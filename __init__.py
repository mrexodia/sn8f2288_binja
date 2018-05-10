from binaryninja.architecture import Architecture
from binaryninja.lowlevelil import LowLevelILLabel, LLIL_TEMP
from binaryninja.function import RegisterInfo, InstructionInfo, InstructionTextToken
from binaryninja.binaryview import BinaryView
from binaryninja.types import Symbol
from binaryninja.log import log_error
from binaryninja.enums import (BranchType, InstructionTextTokenType, Endianness,
                            LowLevelILOperation, LowLevelILFlagCondition, FlagRole, SegmentFlag, SymbolType)

from struct import unpack

def tohex(data):
    return ':'.join(x.encode('hex') for x in data)

NUL_SPACE = 0 # No operand
ZRO_SPACE = 1 # Operand is zero-page ram address
RAM_SPACE = 2 # Operand is ram address
ROM_SPACE = 3 # Operand is rom address
IMM_SPACE = 4 # Operand is immediate value

NONXT = 0 # There is no fixed next-instruction address (RET & RETI).
NEXTI = 1 # Always execute next instruction (normal instructions).
BRNCH = 2 # Execute next instruction or the one after that (conditional branches).
JUMPI = 3 # Jump to some far address (unconditional branches).
CALLI = 4 # Call some address (call instructions).

# opcode: (mask, opspace, jump_action, opcode, [operand])
opcode_dict = {
    0x00: (0x0000, NUL_SPACE, NEXTI, 'NOP',    None),
    0x02: (0x00ff, ZRO_SPACE, NEXTI, 'B0XCH',  ['A', '%s']),
    0x03: (0x00ff, ZRO_SPACE, NEXTI, 'B0ADD',  ['%s', 'A']),
    0x04: (0x0000, NUL_SPACE, NEXTI, 'PUSH',   None),
    0x05: (0x0000, NUL_SPACE, NEXTI, 'POP',    None),
    0x06: (0x00ff, IMM_SPACE, BRNCH, 'CMPRS',  ['A', '%s']),
    0x07: (0x00ff, RAM_SPACE, BRNCH, 'CMPRS',  ['A', '%s']),
    0x08: (0x00ff, RAM_SPACE, NEXTI, 'RRC',    ['%s']),
    0x09: (0x00ff, RAM_SPACE, NEXTI, 'RRCM',   ['%s']),
    0x0a: (0x00ff, RAM_SPACE, NEXTI, 'RLC',    ['%s']),
    0x0b: (0x00ff, RAM_SPACE, NEXTI, 'RLCM',   ['%s']),
    0x0d: (0x0000, NUL_SPACE, NEXTI, 'MOVC',   None),
    0x0e: (0x0000, NUL_SPACE, NONXT, 'RET',    None),
    0x0f: (0x0000, NUL_SPACE, NONXT, 'RETI',   None),
    0x10: (0x00ff, RAM_SPACE, NEXTI, 'ADC',    ['A, %s']),
    0x11: (0x00ff, RAM_SPACE, NEXTI, 'ADC',    ['%s', 'A']),
    0x12: (0x00ff, RAM_SPACE, NEXTI, 'ADD',    ['A', '%s']),
    0x13: (0x00ff, RAM_SPACE, NEXTI, 'ADD',    ['%s', 'A']),
    0x14: (0x00ff, IMM_SPACE, NEXTI, 'ADD',    ['A', '%s']),
    0x15: (0x00ff, RAM_SPACE, BRNCH, 'INCS',   ['%s']),
    0x16: (0x00ff, RAM_SPACE, BRNCH, 'INCMS',  ['%s']),
    0x17: (0x00ff, RAM_SPACE, NEXTI, 'SWAP',   ['%s']),
    0x18: (0x00ff, RAM_SPACE, NEXTI, 'OR',     ['A', '%s']),
    0x19: (0x00ff, RAM_SPACE, NEXTI, 'OR',     ['%s', 'A']),
    0x1a: (0x00ff, IMM_SPACE, NEXTI, 'OR',     ['A', '%s']),
    0x1b: (0x00ff, RAM_SPACE, NEXTI, 'XOR',    ['A', '%s']),
    0x1c: (0x00ff, RAM_SPACE, NEXTI, 'XOR',    ['%s', 'A']),
    0x1d: (0x00ff, IMM_SPACE, NEXTI, 'XOR',    ['A', '%s']),
    0x1e: (0x00ff, RAM_SPACE, NEXTI, 'MOV',    ['A', '%s']),
    0x1f: (0x00ff, RAM_SPACE, NEXTI, 'MOV',    ['%s', 'A']),
    0x20: (0x00ff, RAM_SPACE, NEXTI, 'SBC',    ['A', '%s']),
    0x21: (0x00ff, RAM_SPACE, NEXTI, 'SBC',    ['%s', 'A']),
    0x22: (0x00ff, RAM_SPACE, NEXTI, 'SUB',    ['A', '%s']),
    0x23: (0x00ff, RAM_SPACE, NEXTI, 'SUB',    ['%s', 'A']),
    0x24: (0x00ff, IMM_SPACE, NEXTI, 'SUB',    ['A', '%s']),
    0x25: (0x00ff, RAM_SPACE, BRNCH, 'DECS',   ['%s']),
    0x26: (0x00ff, RAM_SPACE, BRNCH, 'DECMS',  ['%s']),
    0x27: (0x00ff, RAM_SPACE, NEXTI, 'SWAPM',  ['%s']),
    0x28: (0x00ff, RAM_SPACE, NEXTI, 'AND',    ['A', '%s']),
    0x29: (0x00ff, RAM_SPACE, NEXTI, 'AND',    ['%s', 'A']),
    0x2a: (0x00ff, IMM_SPACE, NEXTI, 'AND',    ['A', '%s']),
    0x2b: (0x00ff, RAM_SPACE, NEXTI, 'CLR',    ['%s']),
    0x2c: (0x00ff, RAM_SPACE, NEXTI, 'XCH',    ['A', '%s']),
    0x2d: (0x00ff, IMM_SPACE, NEXTI, 'MOV',    ['A', '%s']),
    0x2e: (0x00ff, ZRO_SPACE, NEXTI, 'B0MOV',  ['A', '%s']),
    0x2f: (0x00ff, RAM_SPACE, NEXTI, 'B0MOV',  ['%s', 'A']),
    0x32: (0x00ff, IMM_SPACE, NEXTI, 'B0MOV',  ['R', '%s']),
    0x33: (0x00ff, IMM_SPACE, NEXTI, 'B0MOV',  ['Z', '%s']),
    0x34: (0x00ff, IMM_SPACE, NEXTI, 'B0MOV',  ['Y', '%s']),
    0x36: (0x00ff, IMM_SPACE, NEXTI, 'B0MOV',  ['PFLAG', '%s']),
    0x37: (0x00ff, IMM_SPACE, NEXTI, 'B0MOV',  ['RBANK', '%s']),
    0x40: (0x00ff, RAM_SPACE, NEXTI, 'BCLR',   ['%s']),
    0x48: (0x00ff, RAM_SPACE, NEXTI, 'BSET',   ['%s']),
    0x50: (0x00ff, RAM_SPACE, BRNCH, 'BTS0',   ['%s']),
    0x58: (0x00ff, RAM_SPACE, BRNCH, 'BTS1',   ['%s']),
    0x60: (0x00ff, ZRO_SPACE, NEXTI, 'B0BCLR', ['%s']),
    0x68: (0x00ff, ZRO_SPACE, NEXTI, 'B0BSET', ['%s']),
    0x70: (0x00ff, ZRO_SPACE, BRNCH, 'B0BTS0', ['%s']),
    0x78: (0x00ff, ZRO_SPACE, BRNCH, 'B0BTS1', ['%s']),
    0x80: (0x3fff, ROM_SPACE, JUMPI, 'JMP',    ['%s']),
    0xc0: (0x3fff, ROM_SPACE, CALLI, 'CALL',   ['%s']),
}

DEFAULT_SYMBOL = (None, {})

def disassemble(address, instruction):
    bincode = instruction >> 8
    if bincode >= 0x80:
        opcode_key = bincode & 0xc0
        is_bit = False
    elif bincode >= 0x40:
        opcode_key = bincode & 0xf8
        is_bit = True
    else:
        opcode_key = bincode
        is_bit = False
    try:
        mask, opspace, jump_action, opcode, operands = opcode_dict[opcode_key]
    except KeyError:
        return None
        
    tokens = []
    
    tokens.append(InstructionTextToken(InstructionTextTokenType.TextToken, '.ORG'))
    tokens.append(InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, ' '))
    tokens.append(InstructionTextToken(InstructionTextTokenType.TextToken, '0x%04X' % address))
    tokens.append(InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, ' '))
    
    tokens.append(InstructionTextToken(InstructionTextTokenType.InstructionToken, opcode))
    
    type = None
    value = 0
    opaddr = 0

    if opspace != NUL_SPACE:
        operand = instruction & mask
        if opspace == ROM_SPACE:
            symbol = None
            operand_fmt = '0x%04x'
            type = InstructionTextTokenType.PossibleAddressToken
            opaddr = operand * 2
            assert not is_bit
        elif opspace == IMM_SPACE:
            symbol = None
            operand_fmt = '#0x%02x'
            type = InstructionTextTokenType.IntegerToken
            value = operand
            assert not is_bit
        else: # ZRO & RAM
            #for start, stop, ram_symbol_dict in ram_range_symbol_list:
            #    if start <= address <= stop and operand in ram_symbol_dict:
            #        symbol, bit_dict = ram_symbol_dict[operand]
            #        break
            #else:
            #    symbol, bit_dict = DEFAULT_SYMBOL
            symbol, bit_dict = DEFAULT_SYMBOL
            operand_fmt = '0x%02x'
            type = InstructionTextTokenType.PossibleAddressToken
            if is_bit:
                bit_address = bincode & 0x7
                bit_symbol = bit_dict.get(bit_address)
                if bit_symbol is None:
                    if symbol is None:
                        operand_fmt += '.%i' % bit_address
                    else:
                        symbol += '.%i' % bit_address
                else:
                    symbol = bit_symbol
            else:
                value = operand
        if symbol is None:
            symbol = operand_fmt % operand
            #if opspace in (ZRO_SPACE, RAM_SPACE):
            #    missing_ram_symbol_dict[symbol].append(address)
        
        tokens.append(InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, ' '))
        i = 0
        for op in operands:
            if i > 0:
                tokens.append(InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, ', '))
            if op == '%s':
                tokens.append(InstructionTextToken(type, symbol, value, address=opaddr))
            else:
                tokens.append(InstructionTextToken(InstructionTextTokenType.RegisterToken, op))
            i += 1
        #opcode += '\t' + caption % symbol

    return tokens

class SN8F2288(Architecture):
    name = "SN8F2288"
    endianness = Endianness.LittleEndian
    address_size = 2
    default_int_size = 1
    instr_alignment = 1
    max_instr_length = 2
    regs = {
        "STKP": RegisterInfo("STKP", 1),
        "A": RegisterInfo("A", 1),
        "R": RegisterInfo("R", 1),
        "Z": RegisterInfo("Z", 1),
        "Y": RegisterInfo("Y", 1),
        "PFLAG": RegisterInfo("PFLAG", 1),
        "RBANK": RegisterInfo("RBANK", 1)
    }
    stack_pointer = "STKP"
    flags = []
    flag_write_types = []
    flag_roles = {}
    flags_required_for_flag_condition = {}
    flags_written_by_flag_write_type = {}

    def perform_get_instruction_info(self, data, addr):
        if len(data) < 2:
            print "perform_get_instruction_info(%s, 0x%04x), not enough data!" % (tohex(data), addr)
            return None
        if addr % 2 != 0:
            print "perform_get_instruction_info(%s, 0x%04x), address not aligned!" % (tohex(data), addr)
            return None
        #print "perform_get_instruction_info(%s, 0x%04x)" % (tohex(data), addr)
        info = InstructionInfo()
        info.length = self.max_instr_length

        # workaround for a Binary Ninja bug, data is not guaranteed to be max_instr_length bytes
        data = data[:self.max_instr_length]
        
        instruction = unpack('<H', data)[0]
        bincode = instruction >> 8
        if bincode >= 0x80:
            opcode_key = bincode & 0xc0
            is_bit = False
        elif bincode >= 0x40:
            opcode_key = bincode & 0xf8
            is_bit = True
        else:
            opcode_key = bincode
            is_bit = False
        try:
            mask, opspace, jump_action, opcode, caption = opcode_dict[opcode_key]
        except KeyError:
            return None # TODO is it possible to get more information?
            
        operand = instruction & mask
        branches = {
            NONXT: lambda: [(BranchType.FunctionReturn, 0)],
            NEXTI: lambda: [],
            BRNCH: lambda: [(BranchType.TrueBranch, addr + 4), (BranchType.FalseBranch, addr + 2)],
            JUMPI: lambda: [(BranchType.UnconditionalBranch, operand * 2)], # ROM addresses are 16 bits of data per address
            CALLI: lambda: [(BranchType.CallDestination, operand * 2)] # ROM addresses are 16 bits of data per address
        }[jump_action]()
        
        for type, address in branches:
            info.add_branch(type, address)
        return info

    def perform_get_instruction_text(self, data, addr):
        if len(data) < 2:
            print "perform_get_instruction_text(%s, 0x%04x), not enough data!" % (tohex(data), addr)
            return None
        if addr % 2 != 0:
            print "perform_get_instruction_text(%s, 0x%04x), address not aligned!" % (tohex(data), addr)
            return None, None
        #print "perform_get_instruction_text(%s, 0x%04x)" % (tohex(data), addr)
        
        # workaround for a Binary Ninja bug, data is not guaranteed to be max_instr_length bytes
        data = data[:self.max_instr_length]

        instruction = unpack('<H', data)[0]
        tokens = disassemble(addr / 2, instruction)
        #tokens = []
        #tokens.append(InstructionTextToken(InstructionTextTokenType.TextToken, ".ORG"))
        #tokens.append(InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, " "))
        #tokens.append(InstructionTextToken(InstructionTextTokenType.TextToken, "0x%04X" % (addr / 2)))
        #tokens.append(InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, " "))
        #tokens.append(InstructionTextToken(InstructionTextTokenType.InstructionToken, "DW"))
        #tokens.append(InstructionTextToken(InstructionTextTokenType.OperandSeparatorToken, " "))
        #tokens.append(InstructionTextToken(InstructionTextTokenType.HexDumpTextToken, "0x%04X" % instruction))
        return tokens, self.max_instr_length

    def perform_get_instruction_low_level_il(self, data, addr, il):
        return None

SN8F2288.register()
def sg_symbol_to_number(sg_symbol):
    sg_dict = {
        'P1': 1, 
        'P-1': 2,
        'P2': 3,
        'P21': 4,
        'C2': 5,
        'Pm': 6, 
        'Pc': 7, 
        'Cm': 8, 
        'Cc': 9, 
        'P2/m': 10,
        'P21/m': 11, 
        'C2/m': 12, 
        'P2/c': 13, 
        'P21/c': 14, 
        'C2/c': 15,
        'P222': 16, 
        'P2221': 17, 
        'P21212': 18, 
        'P212121': 19, 
        'C2221': 20,
        'C222': 21, 
        'F222': 22, 
        'I222': 23, 
        'I212121': 24, 
        'Pmm2': 25,
        'Pmc21': 26, 
        'Pcc2': 27, 
        'Pma2': 28, 
        'Pca21': 29, 
        'Pnc2': 30,
        'Pmn21': 31, 
        'Pba2': 32,
        'Pna21': 33,
        'Pnn2': 34,
        'Cmm2': 35,
        'Cmc21': 36,
        'Ccc2': 37,
        'Amm2': 38,
        'Abm2': 39, # Aem2?
        'Ama2': 40,
        'Aba2': 41, # Aea2?
        'Fmm2': 42,
        'Fdd2': 43,
        'Imm2': 44,
        'Iba2': 45,
        'Ima2': 46,
        'Pmmm': 47,
        'Pnnn': 48,
        'Pccm': 49,
        'Pban': 50,
        'Pmma': 51,
        'Pnna': 52,
        'Pmna': 53,
        'Pcca': 54,
        'Pbam': 55,
        'Pccn': 56,
        'Pbcm': 57,
        'Pnnm': 58,
        'Pmmn': 59,
        'Pbcn': 60,
        'Pbca': 61,
        'Pnma': 62,
        'Cmcm': 63,
        'Cmca': 64, # Cmce?
        'Cmmm': 65,
        'Cccm': 66,
        'Cmma': 67, # Cmme?
        'Ccca': 68, # Ccce?
        'Fmmm': 69,
        'Fddd': 70,
        'Immm': 71,
        'Ibam': 72,
        'Ibca': 73,
        'Imma': 74,
        'P4': 75,
        'P41': 76,
        'P42': 77,
        'P43': 78,
        'I4': 79,
        'I41': 80,
        'P-4': 81,
        'I-4': 82,
        'P4/m': 83,
        'P42/m': 84,
        'P4/n': 85,
        'P42/n': 86,
        'I4/m': 87,
        'I41/a': 88,
        'P422': 89,
        'P4212': 90,
        'P4122': 91,
        'P41212': 92,
        'P4222': 93,
        'P42212': 94,
        'P4322': 95,
        'P43212': 96,
        'I422': 97,
        'I4122': 98,
        'P4mm': 99,
        'P4bm': 100,
        'P42cm': 101,
        'P42nm': 102,
        'P4cc': 103,
        'P4nc': 104,
        'P42mc': 105,
        'P42bc': 106,
        'I4mm': 107,
        'I4cm': 108,
        'I41md': 109,
        'I41cd': 110,
        'P-42m': 111,
        'P-42c': 112,
        'P-421m': 113,
        'P-421c': 114,
        'P-4m2': 115,
        'P-4c2': 116,
        'P-4b2': 117,
        'P-4n2': 118,
        'I-4m2': 119,
        'I-4c2': 120,
        'I-42m': 121,
        'I-42d': 122,
        'P4/mmm': 123,
        'P4/mcc': 124,
        'P4/nbm': 125,
        'P4/nnc': 126,
        'P4/mbm': 127,
        'P4/mnc': 128,
        'P4/nmm': 129,
        'P4/ncc': 130,
        'P42/mmc': 131,
        'P42/mcm': 132,
        'P42/nbc': 133,
        'P42/nnm': 134,
        'P42/mbc': 135,
        'P42/mnm': 136,
        'P42/nmc': 137,
        'P42/ncm': 138,
        'I4/mmm': 139,
        'I4/mcm': 140,
        'I41/amd': 141,
        'I41/acd': 142,
        'P3': 143,
        'P31': 144,
        'P32': 145,
        'R3': 146,
        'P-3': 147,
        'R-3': 148,
        'P312': 149,
        'P321': 150,
        'P3112': 151,
        'P3121': 152,
        'P3212': 153,
        'P3221': 154,
        'R32': 155,
        'P3m1': 156,
        'P31m': 157,
        'P3c1': 158,
        'P31c': 159,
        'R3m': 160,
        'R3c': 161,
        'P-31m': 162,
        'P-31c': 163,
        'P-3m1': 164,
        'P-3c1': 165,
        'R-3m': 166,
        'R-3c': 167,
        'P6': 168,
        'P61': 169,
        'P65': 170,
        'P62': 171,
        'P64': 172,
        'P63': 173,
        'P-6': 174,
        'P6/m': 175,
        'P63/m': 176,
        'P622': 177,
        'P6122': 178,
        'P6522': 179,
        'P6222': 180,
        'P6422': 181,
        'P6322': 182,
        'P6mm': 183,
        'P6cc': 184,
        'P63cm': 185,
        'P63mc': 186,
        'P-6m2': 187,
        'P-6c2': 188,
        'P-62m': 189,
        'P-62c': 190,
        'P6/mmm': 191,
        'P6/mcc': 192,
        'P63/mcm': 193,
        'P63/mmc': 194,
        'P23': 195,
        'F23': 196,
        'I23': 197,
        'P213': 198,
        'I213': 199,
        'Pm-3': 200,
        'Pn-3': 201,
        'Fm-3': 202,
        'Fd-3': 203,
        'Im-3': 204,
        'Pa-3': 205,
        'Ia-3': 206,
        'P432': 207,
        'P4232': 208,
        'F432': 209,
        'F4132': 210,
        'I432': 211,
        'P4332': 212,
        'P4132': 213,
        'I4132': 214,
        'P-43m': 215,
        'F-43m': 216,
        'I-43m': 217,
        'P-43n': 218,
        'F-43c': 219,
        'I-43d': 220,
        'Pm-3m': 221,
        'Pn-3n': 222,
        'Pm-3n': 223,
        'Pn-3m': 224,
        'Fm-3m': 225,
        'Fm-3c': 226,
        'Fd-3m': 227,
        'Fd-3c': 228,
        'Im-3m': 229,
        'Ia-3d': 230
    }
    if sg_symbol in sg_dict:
        sg_number = sg_dict[sg_symbol]
        return sg_number
    else:
        return None


def find_wyckoff_pos(sg_number):
    if sg_number == 1:
        wyckoff_pos = '1a'
    elif sg_number == 2:
        wyckoff_pos = '2i'
    elif sg_number == 3:
        wyckoff_pos = '2e'
    elif sg_number == 4:
        wyckoff_pos = '2a'
    elif sg_number == 5:
        wyckoff_pos = '4c'
    elif sg_number == 6:
        wyckoff_pos = '2c'
    elif sg_number == 7:
        wyckoff_pos = '2a'
    elif sg_number == 8:
        wyckoff_pos = '4b'
    elif sg_number == 9:
        wyckoff_pos = '4a'
    elif sg_number == 10:
        wyckoff_pos = '4o'
    elif sg_number == 11:
        wyckoff_pos = '4f'
    elif sg_number == 12:
        wyckoff_pos = '8j'
    elif sg_number == 13:
        wyckoff_pos = '4g'
    elif sg_number == 14:
        wyckoff_pos = '4e'
    elif sg_number == 15:
        wyckoff_pos = '8f'
    elif sg_number == 16:
        wyckoff_pos = '4u'
    elif sg_number == 17:
        wyckoff_pos = '4e'
    elif sg_number == 18:
        wyckoff_pos = '4c'
    elif sg_number == 19:
        wyckoff_pos = '4a'
    elif sg_number == 20:
        wyckoff_pos = '8c'
    elif sg_number == 21:
        wyckoff_pos = '8l'
    elif sg_number == 22:
        wyckoff_pos = '16k'
    elif sg_number == 23:
        wyckoff_pos = '8k'
    elif sg_number == 24:
        wyckoff_pos = '8d'
    elif sg_number == 25:
        wyckoff_pos = '4i'
    elif sg_number == 26:
        wyckoff_pos = '4c'
    elif sg_number == 27:
        wyckoff_pos = '4e'
    elif sg_number == 28:
        wyckoff_pos = '4d'
    elif sg_number == 29:
        wyckoff_pos = '4a'
    elif sg_number == 30:
        wyckoff_pos = '4c'
    elif sg_number == 31:
        wyckoff_pos = '4b'
    elif sg_number == 32:
        wyckoff_pos = '4c'
    elif sg_number == 33:
        wyckoff_pos = '4a'
    elif sg_number == 34:
        wyckoff_pos = '4c'
    elif sg_number == 35:
        wyckoff_pos = '8f'
    elif sg_number == 36:
        wyckoff_pos = '8b'
    elif sg_number == 37:
        wyckoff_pos = '8d'
    elif sg_number == 38:
        wyckoff_pos = '8f'
    elif sg_number == 39:
        wyckoff_pos = '8d'
    elif sg_number == 40:
        wyckoff_pos = '8c'
    elif sg_number == 41:
        wyckoff_pos = '8b'
    elif sg_number == 42:
        wyckoff_pos = '16e'
    elif sg_number == 43:
        wyckoff_pos = '16b'
    elif sg_number == 44:
        wyckoff_pos = '8e'
    elif sg_number == 45:
        wyckoff_pos = '8c'
    elif sg_number == 46:
        wyckoff_pos = '8c'
    elif sg_number == 47:
        wyckoff_pos = '8A'
    elif sg_number == 48:
        wyckoff_pos = '8m'
    elif sg_number == 49:
        wyckoff_pos = '8r'
    elif sg_number == 50:
        wyckoff_pos = '8m'
    elif sg_number == 51:
        wyckoff_pos = '8l'
    elif sg_number == 52:
        wyckoff_pos = '8e'
    elif sg_number == 53:
        wyckoff_pos = '8i'
    elif sg_number == 54:
        wyckoff_pos = '8f'
    elif sg_number == 55:
        wyckoff_pos = '8i'
    elif sg_number == 56:
        wyckoff_pos = '8e'
    elif sg_number == 57:
        wyckoff_pos = '8e'
    elif sg_number == 58:
        wyckoff_pos = '8h'
    elif sg_number == 59:
        wyckoff_pos = '8g'
    elif sg_number == 60:
        wyckoff_pos = '8d'
    elif sg_number == 61:
        wyckoff_pos = '8c'
    elif sg_number == 62:
        wyckoff_pos = '8d'
    elif sg_number == 63:
        wyckoff_pos = '16h'
    elif sg_number == 64:
        wyckoff_pos = '16g'
    elif sg_number == 65:
        wyckoff_pos = '16r'
    elif sg_number == 66:
        wyckoff_pos = '16m'
    elif sg_number == 67:
        wyckoff_pos = '16o'
    elif sg_number == 68:
        wyckoff_pos = '16i'
    elif sg_number == 69:
        wyckoff_pos = '32p'
    elif sg_number == 70:
        wyckoff_pos = '32h'
    elif sg_number == 71:
        wyckoff_pos = '16o'
    elif sg_number == 72:
        wyckoff_pos = '16k'
    elif sg_number == 73:
        wyckoff_pos = '16f'
    elif sg_number == 74:
        wyckoff_pos = '16j'
    elif sg_number == 75:
        wyckoff_pos = '4d'
    elif sg_number == 76:
        wyckoff_pos = '4a'
    elif sg_number == 77:
        wyckoff_pos = '4d'
    elif sg_number == 78:
        wyckoff_pos = '4a'
    elif sg_number == 79:
        wyckoff_pos = '8c'
    elif sg_number == 80:
        wyckoff_pos = '8b'
    elif sg_number == 81:
        wyckoff_pos = '4h'
    elif sg_number == 82:
        wyckoff_pos = '8g'
    elif sg_number == 83:
        wyckoff_pos = '8l'
    elif sg_number == 84:
        wyckoff_pos = '8k'
    elif sg_number == 85:
        wyckoff_pos = '8g'
    elif sg_number == 86:
        wyckoff_pos = '8g'
    elif sg_number == 87:
        wyckoff_pos = '16i'
    elif sg_number == 88:
        wyckoff_pos = '16f'
    elif sg_number == 89:
        wyckoff_pos = '8p'
    elif sg_number == 90:
        wyckoff_pos = '8g'
    elif sg_number == 91:
        wyckoff_pos = '8d'
    elif sg_number == 92:
        wyckoff_pos = '8b'
    elif sg_number == 93:
        wyckoff_pos = '8p'
    elif sg_number == 94:
        wyckoff_pos = '8g'
    elif sg_number == 95:
        wyckoff_pos = '8d'
    elif sg_number == 96:
        wyckoff_pos = '8b'
    elif sg_number == 97:
        wyckoff_pos = '16k'
    elif sg_number == 98:
        wyckoff_pos = '16g'
    elif sg_number == 99:
        wyckoff_pos = '8g'
    elif sg_number == 100:
        wyckoff_pos = '8d'
    elif sg_number == 101:
        wyckoff_pos = '8e'
    elif sg_number == 102:
        wyckoff_pos = '8d'
    elif sg_number == 103:
        wyckoff_pos = '8d'
    elif sg_number == 104:
        wyckoff_pos = '8c'
    elif sg_number == 105:
        wyckoff_pos = '8f'
    elif sg_number == 106:
        wyckoff_pos = '8c'
    elif sg_number == 107:
        wyckoff_pos = '16e'
    elif sg_number == 108:
        wyckoff_pos = '16d'
    elif sg_number == 109:
        wyckoff_pos = '16c'
    elif sg_number == 110:
        wyckoff_pos = '16b'
    elif sg_number == 111:
        wyckoff_pos = '8o'
    elif sg_number == 112:
        wyckoff_pos = '8n'
    elif sg_number == 113:
        wyckoff_pos = '8f'
    elif sg_number == 114:
        wyckoff_pos = '8e'
    elif sg_number == 115:
        wyckoff_pos = '8l'
    elif sg_number == 116:
        wyckoff_pos = '8j'
    elif sg_number == 117:
        wyckoff_pos = '8i'
    elif sg_number == 118:
        wyckoff_pos = '8i'
    elif sg_number == 119:
        wyckoff_pos = '16j'
    elif sg_number == 120:
        wyckoff_pos = '16i'
    elif sg_number == 121:
        wyckoff_pos = '16j'
    elif sg_number == 122:
        wyckoff_pos = '16e'
    elif sg_number == 123:
        wyckoff_pos = '16u'
    elif sg_number == 124:
        wyckoff_pos = '16n'
    elif sg_number == 125:
        wyckoff_pos = '16n'
    elif sg_number == 126:
        wyckoff_pos = '16k'
    elif sg_number == 127:
        wyckoff_pos = '16l'
    elif sg_number == 128:
        wyckoff_pos = '16i'
    elif sg_number == 129:
        wyckoff_pos = '16k'
    elif sg_number == 130:
        wyckoff_pos = '16g'
    elif sg_number == 131:
        wyckoff_pos = '16r'
    elif sg_number == 132:
        wyckoff_pos = '16p'
    elif sg_number == 133:
        wyckoff_pos = '16k'
    elif sg_number == 134:
        wyckoff_pos = '16n'
    elif sg_number == 135:
        wyckoff_pos = '16i'
    elif sg_number == 136:
        wyckoff_pos = '16k'
    elif sg_number == 137:
        wyckoff_pos = '16h'
    elif sg_number == 138:
        wyckoff_pos = '16j'
    elif sg_number == 139:
        wyckoff_pos = '32o'
    elif sg_number == 140:
        wyckoff_pos = '32m'
    elif sg_number == 141:
        wyckoff_pos = '32i'
    elif sg_number == 142:
        wyckoff_pos = '32g'
    elif sg_number == 143:
        wyckoff_pos = '3d'
    elif sg_number == 144:
        wyckoff_pos = '3a'
    elif sg_number == 145:
        wyckoff_pos = '3a'
    elif sg_number == 146:
        wyckoff_pos = '9b'
    elif sg_number == 147:
        wyckoff_pos = '6g'
    elif sg_number == 148:
        wyckoff_pos = '18f'
    elif sg_number == 149:
        wyckoff_pos = '6l'
    elif sg_number == 150:
        wyckoff_pos = '6g'
    elif sg_number == 151:
        wyckoff_pos = '6c'
    elif sg_number == 152:
        wyckoff_pos = '6c'
    elif sg_number == 153:
        wyckoff_pos = '6c'
    elif sg_number == 154:
        wyckoff_pos = '6c'
    elif sg_number == 155:
        wyckoff_pos = '18f'
    elif sg_number == 156:
        wyckoff_pos = '6e'
    elif sg_number == 157:
        wyckoff_pos = '6d'
    elif sg_number == 158:
        wyckoff_pos = '6d'
    elif sg_number == 159:
        wyckoff_pos = '6c'
    elif sg_number == 160:
        wyckoff_pos = '18c'
    elif sg_number == 161:
        wyckoff_pos = '18b'
    elif sg_number == 162:
        wyckoff_pos = '12l'
    elif sg_number == 163:
        wyckoff_pos = '12i'
    elif sg_number == 164:
        wyckoff_pos = '12j'
    elif sg_number == 165:
        wyckoff_pos = '12g'
    elif sg_number == 166:
        wyckoff_pos = '36i'
    elif sg_number == 167:
        wyckoff_pos = '36f'
    elif sg_number == 168:
        wyckoff_pos = '6d'
    elif sg_number == 169:
        wyckoff_pos = '6a'
    elif sg_number == 170:
        wyckoff_pos = '6a'
    elif sg_number == 171:
        wyckoff_pos = '6c'
    elif sg_number == 172:
        wyckoff_pos = '6c'
    elif sg_number == 173:
        wyckoff_pos = '6c'
    elif sg_number == 174:
        wyckoff_pos = '6l'
    elif sg_number == 175:
        wyckoff_pos = '12l'
    elif sg_number == 176:
        wyckoff_pos = '12i'
    elif sg_number == 177:
        wyckoff_pos = '12n'
    elif sg_number == 178:
        wyckoff_pos = '12c'
    elif sg_number == 179:
        wyckoff_pos = '12c'
    elif sg_number == 180:
        wyckoff_pos = '12k'
    elif sg_number == 181:
        wyckoff_pos = '12k'
    elif sg_number == 182:
        wyckoff_pos = '12i'
    elif sg_number == 183:
        wyckoff_pos = '12f'
    elif sg_number == 184:
        wyckoff_pos = '12d'
    elif sg_number == 185:
        wyckoff_pos = '12d'
    elif sg_number == 186:
        wyckoff_pos = '12d'
    elif sg_number == 187:
        wyckoff_pos = '12o'
    elif sg_number == 188:
        wyckoff_pos = '12l'
    elif sg_number == 189:
        wyckoff_pos = '12l'
    elif sg_number == 190:
        wyckoff_pos = '12i'
    elif sg_number == 191:
        wyckoff_pos = '24r'
    elif sg_number == 192:
        wyckoff_pos = '24m'
    elif sg_number == 193:
        wyckoff_pos = '24l'
    elif sg_number == 194:
        wyckoff_pos = '24l'
    elif sg_number == 195:
        wyckoff_pos = '12j'
    elif sg_number == 196:
        wyckoff_pos = '48h'
    elif sg_number == 197:
        wyckoff_pos = '24f'
    elif sg_number == 198:
        wyckoff_pos = '12b'
    elif sg_number == 199:
        wyckoff_pos = '24c'
    elif sg_number == 200:
        wyckoff_pos = '24l'
    elif sg_number == 201:
        wyckoff_pos = '24h'
    elif sg_number == 202:
        wyckoff_pos = '96i'
    elif sg_number == 203:
        wyckoff_pos = '96g'
    elif sg_number == 204:
        wyckoff_pos = '48h'
    elif sg_number == 205:
        wyckoff_pos = '24d'
    elif sg_number == 206:
        wyckoff_pos = '48e'
    elif sg_number == 207:
        wyckoff_pos = '24k'
    elif sg_number == 208:
        wyckoff_pos = '24m'
    elif sg_number == 209:
        wyckoff_pos = '96j'
    elif sg_number == 210:
        wyckoff_pos = '96h'
    elif sg_number == 211:
        wyckoff_pos = '48j'
    elif sg_number == 212:
        wyckoff_pos = '24e'
    elif sg_number == 213:
        wyckoff_pos = '24e'
    elif sg_number == 214:
        wyckoff_pos = '48i'
    elif sg_number == 215:
        wyckoff_pos = '24j'
    elif sg_number == 216:
        wyckoff_pos = '96i'
    elif sg_number == 217:
        wyckoff_pos = '48h'
    elif sg_number == 218:
        wyckoff_pos = '24i'
    elif sg_number == 219:
        wyckoff_pos = '96h'
    elif sg_number == 220:
        wyckoff_pos = '48e'
    elif sg_number == 221:
        wyckoff_pos = '48n'
    elif sg_number == 222:
        wyckoff_pos = '48i'
    elif sg_number == 223:
        wyckoff_pos = '48l'
    elif sg_number == 224:
        wyckoff_pos = '48l'
    elif sg_number == 225:
        wyckoff_pos = '192l'
    elif sg_number == 226:
        wyckoff_pos = '192j'
    elif sg_number == 227:
        wyckoff_pos = '192i'
    elif sg_number == 228:
        wyckoff_pos = '192h'
    elif sg_number == 229:
        wyckoff_pos = '96l'
    elif sg_number == 230:
        wyckoff_pos = '96h'
    return wyckoff_pos


def check_chiral(sg_symbol):
    chiral_sg_list = ['P1', 'P2', 'P21', 'C2', 'P222', 'P2221', 'P21212', 'P212121', 'C2221',
                     'C222', 'F222', 'I222', 'I212121', 'P4', 'P41', 'P42', 'P43', 'I4', 'I41',
                     'P422', 'P4212', 'P4122', 'P41212', 'P4222', 'P42212', 'P4322', 'P43212',
                     'I422', 'I4122', 'P3', 'P31', 'P32', 'R3', 'P312', 'P321', 'P3112', 'P3121',
                     'P3212', 'P3221', 'R32', 'P6', 'P61', 'P65', 'P63', 'P62', 'P64', 'P622',
                     'P6122', 'P6522', 'P6222', 'P6422', 'P6322', 'P23', 'F23', 'I23', 'P213',
                     'I213', 'P432', 'P4232', 'F432', 'F4132', 'I432', 'P4332', 'P4132', 'I4132']
    if sg_symbol in chiral_sg_list:
        return 1
    else:
        return 0


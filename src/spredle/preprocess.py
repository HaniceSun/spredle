import math

class DataProcessor:
    def __init__(self):
        pass
    def cut_seq(self, in_file, nt=5000, flank=5000):
        out_file = f'{in_file.split(".txt")[0]}_nt{nt}_flank{flank}.txt'
        inFile = open(in_file)
        ouFile = open(out_file, 'w')
        ouFile.write('\t'.join(['gene', 'ch', 'strand', 'start', 'end', 'X', 'y']) + '\n')
        for line in inFile:
            line = line.strip()
            fields = line.split('\t')
            gene = fields[12] + '|' + fields[13]
            ch = fields[0]
            strand = fields[5]
            tx_start = str(int(fields[1]) + 1)
            tx_end = fields[2]
            tx_seq = fields[-2]
            tx_seq_y = fields[-1]

            tx_seq_pad = tx_seq + 'N' * (math.ceil(len(tx_seq)/nt)*nt-len(tx_seq))
            tx_seq_y_pad = tx_seq_y + 'N' * (math.ceil(len(tx_seq_y)/nt)*nt-len(tx_seq_y))
            for n in range(0, len(tx_seq_pad), nt):
                start = n - flank
                end = n + nt + flank
                left = ''
                right = ''
                if start < 0:
                    left = 'N' * abs(start)
                    start = 0
                if end > len(tx_seq_pad):
                    right = 'N' * (end - len(tx_seq_pad))
                    end = len(tx_seq_pad)
                X = left + tx_seq_pad[start:end] + right
                y = tx_seq_y_pad[n:n+nt]
                ouFile.write('\t'.join([gene, ch, strand, tx_start, tx_end, X, y]) + '\n')
        inFile.close()
        ouFile.close()

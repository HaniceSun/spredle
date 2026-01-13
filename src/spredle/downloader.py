import os
import gzip
import subprocess

class DataDownloader:
    def __init__(self):
        self.chrs = ['chr' + str(x) for x in range(1, 23)] + ['chrX', 'chrY']
        self.gene_filter = ['Ensembl_canonical', 'protein_coding']

    def download_training_data(self, genome_reference=None, gene_annotation=None):
        if not genome_reference:
            genome_reference = 'https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz'

        if not gene_annotation: 
            gene_annotation = 'https://ftp.ensembl.org/pub/current_gtf/homo_sapiens/Homo_sapiens.GRCh38.115.gtf.gz'

        self.genome_file = genome_reference.split('/')[-1]
        self.gene_file = gene_annotation.split('/')[-1] 


        cmd = f'wget {genome_reference}; wget {gene_annotation}; gunzip {self.gene_file}'
        subprocess.run(cmd, shell=True, check=True)

        self.fasta_concat(self.genome_file)
        self.gtf_to_bed12(self.gene_file.split('.gz')[0])
        self.get_seq_from_reference_bed12()

    def gtf_to_bed12(self, gtf):
        bed12 = gtf.replace('.gtf', '.bed12')
        cmd = f'bedparse gtf2bed {gtf} --extraFields gene_id,gene_name,gene_biotype,transcript_id,transcript_name,transcript_biotype,tag >{bed12}'
        subprocess.run(cmd, shell=True, check=True)

    def get_seq_from_reference_bed12(self, trans=''.maketrans('atcgATCG', 'tagcTAGC')):
        FA = {}
        in_file = self.genome_file.replace('.fa.gz', '.OneLine.fa.gz')
        inFile = gzip.open(in_file, 'rt')
        while True:
            line1 = inFile.readline().strip()
            line2 = inFile.readline().strip()
            if line1:
                ch = 'chr' + line1.split()[0][1:]
                if ch in self.chrs:
                    FA[ch] = '.' + line2.upper()
            else:
                break
        inFile.close()

        ouFile = open(self.gene_file.replace(".gtf.gz", "_seq.txt"), 'w')
        inFile = open(self.gene_file.replace(".gtf.gz", ".bed12"))
        for line in inFile:
            line = line.strip()
            fields = line.split('\t')
            gene = fields[12] + '_' + fields[13]
            ch = fields[0]
            ch_prefix = ''
            if ch.find('chr') == -1:
                ch = 'chr' + ch
                ch_prefix = 'chr'
            strand = fields[5]
            tx_start = int(fields[1]) + 1
            tx_end = int(fields[2])
            exon_start = []
            exon_end = []
            blockStarts = fields[11].split(',')[0:-1]
            blockSizes = fields[10].split(',')[0:-1]
            for n in range(len(blockStarts)):
                exon_start.append(int(fields[1]) + 1 + int(blockStarts[n]))
                exon_end.append(int(fields[1]) + int(blockStarts[n]) + int(blockSizes[n]))

            flag = True
            if fields[18].find(self.gene_filter[0]) == -1 or fields[14].find(self.gene_filter[1]) == -1 or ch not in self.chrs or ch not in FA:
                flag = False
            if flag and len(exon_start) > 1:
                if strand == '+':
                    # donor:2, acceptor:1
                    tx_seq = FA[ch][tx_start:tx_end+1]
                    tx_seq_y = [0] * len(tx_seq)
                    for x in exon_start[1:]:
                        tx_seq_y[x - tx_start] = 1
                    for x in exon_end[0:-1]:
                        tx_seq_y[x - tx_start] = 2
                if strand == '-':
                    tx_seq = FA[ch][tx_start:tx_end+1]
                    tx_seq_y = [0] * len(tx_seq)
                    for x in exon_start[1:]:
                        tx_seq_y[x - tx_start] = 2
                    for x in exon_end[0:-1]:
                        tx_seq_y[x - tx_start] = 1
                    tx_seq = tx_seq[::-1].translate(trans)
                    tx_seq_y = tx_seq_y[::-1]
                tx_seq_y = ''.join([str(x) for x in tx_seq_y])
                ouFile.write(ch_prefix + line + '\t' + ','.join([str(x) for x in exon_start]) + '\t' + ','.join([str(x) for x in exon_end]) + '\t' + tx_seq + '\t' + tx_seq_y + '\n')
        inFile.close()
        ouFile.close()

    def cut_seq(self, in_file, nt=5000, flank_size=5000):
        out_file = f'{in_file.split(".txt")[0]}_nt{nt}_flank{flank_size}.txt'
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
                start = n - flank_size
                end = n + nt + flank_size
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

    def fasta_concat(self, in_file=None):
        try:
            print(f'Processing {in_file} ...')
            self._OneLine(in_file)
        except Exception as e:
            print(f'Error processing {in_file}: {e}')

    def _OneLine(self, in_file):
        out_file = in_file.split('.fa.gz')[0] + '.OneLine.fa.gz'
        ouFile = gzip.open(out_file, 'wt')
        D = {}
        L = []

        inFile = gzip.open(in_file, 'rt')
        for line in inFile:
            line = line.strip()
            if line.find('>') == 0:
                k = line
                D.setdefault(k, [])
                L.append(k)
            else:
                D[k].append(line)
        inFile.close()

        for k in L:
            seq = ''.join(D[k])
            ouFile.write(k + '\n')
            ouFile.write(seq + '\n')
        ouFile.close()

if __name__ == '__main__':
    dd = DataDownloader()
    dd.download_training_data()

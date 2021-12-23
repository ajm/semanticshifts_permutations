from sys import stderr, exit, argv
import numpy as np
from statsmodels.stats.multitest import multipletests


def main() :
    if len(argv) != 3 :
        print("Usage: {} <wordlist> <results>\n".format(argv[0]))
        exit(1)

    wordlist = argv[1]
    results = argv[2]

    # read ground truth shifts
    shifts = {}
    with open(wordlist) as f :
        f.readline() # header
        for line in f :
            line = line.strip()
            if not line : continue
            data = line.split(',')
            shifts[data[1]] = float(data[2])

    data = []
    with open(results) as f :
        f.readline()
        for line in f :
            line = line.strip()
            if not line : continue
            word, freq1, freq2, dist, ppval = line.split()

            if word in shifts :
                data.append((word, freq1, freq2, dist, ppval))
    
    pvalues = np.array([ float(d[4]) for d in data ])
    _,fdr,_,_ = multipletests(pvalues, method='fdr_bh')
    fdr = fdr.tolist()

    print("word freq1 freq2 shift dist pval fdr")
    for idx,d in enumerate(data) :
        word, freq1, freq2, dist, ppval = d
        print(word, freq1, freq2, shifts[word], dist, ppval, fdr[idx])

    print("processed {} words\n".format(len(data)))

    return 0

if __name__ == '__main__' :
    try :
        exit(main())
    except KeyboardInterrupt :
        print("Killed by User\n", file=stderr)
        exit(1)


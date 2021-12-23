from sys import stderr, exit, argv
import re, json, ast, pickle, string, bz2
import numpy as np
from nltk.tokenize import sent_tokenize


def read_LiverpoolFC(fname) :
    with bz2.open(fname,'rt') as f :
        for line in f :
            line = line.strip()
            #tmp = json.loads(line) # file uses illegal single quotes!
            try :
                tmp = ast.literal_eval(line) # this feels wrong...
            except :
                try :
                    tmp = ast.literal_eval("%s \"}" % line)
                except :
                    try :
                        tmp = ast.literal_eval("%s '}" % line)
                    except :
                        #print(line, file=stderr)
                        try :
                            tmp = ast.literal_eval("%s }" % line[:-5])
                        except :
                            continue

            if ('author' not in tmp) or ('body' not in tmp) :
                continue

            if tmp['author'] in ('TweetsInCommentsBot','RemindMeBot','TwitterToStreamable') :
                continue

            if tmp['body'] in ('[deleted]','[removed]','') :
                continue

            yield tmp['body']

def _remove_guff(s) :
    url='http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    markup='\[([^\[\]]+)\]\([^\(\)]+\)'
    tmp = re.sub(markup, r'\1', s)
    tmp = re.sub(url, '', tmp)
    for pat,rep in (('\.+','.'), ('\-+','-'), ('\?+','?'), ('!+','!')) :
        tmp = re.sub(pat, rep, tmp)
    tmp = '\n'.join([ line for line in tmp.split('\n') if not line.startswith('>') ]) # remove quoted text
    return tmp

def process_LiverpoolFC(fname) :
    print("processing {} ...".format(fname))

    for text in read_LiverpoolFC(fname) :
        text = text.strip().lower().replace("\n"," ")
        text = _remove_guff(text)
        text = " ".join(text.strip().split())
        if not text :
            continue

        yield text

def main() :
    good = set(string.ascii_lowercase + string.digits + ' ')
    #f = open("./LiverpoolFC_dataset/LiverpoolFC_ALL_CLEAN.txt", "w")
    for YEAR in ('13','17') :
        f = bz2.open("./data/LiverpoolFC_{}_CLEAN_nopunctuation.txt.bz2".format(YEAR), "wt")
        for text in process_LiverpoolFC('./data/LiverpoolFC_{}.txt.bz2'.format(YEAR)) :
            for sent in sent_tokenize(text) :
                sent = "".join([ c if c in good else '' for c in sent ])
                print(sent, file=f)
            print("", file=f)
        f.close()
    f.close()

    return 0

if __name__ == '__main__' :
    try :
        exit(main())

    except KeyboardInterrupt :
        print("\nKilled by User\n", file=stderr)
        exit(1)


# sub-word embeddings
# 1. average all sub-words
# 2. just use the last sub-word (because it is context sensitive)


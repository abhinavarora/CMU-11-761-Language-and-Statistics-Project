import pexpect

BIN_PATH  = ''
TEXT_FILE = 'buffer.txt'
EVAL_SCRIPT = BIN_PATH + 'evallm'
MODEL_4G = 'dat4g.binlm'
MODEL_3G = 'dat3g.binlm'
MODEL_5G = 'dat5g.binlm'
MODEL_6G = 'dat6g.binlm'
MODEL_7G = 'dat7g.binlm'
MODEL_PATH = ''

EVAL_COMMAND = './{} -binary {}'
PERP_COMMAND = 'perplexity -text {}'

NtoM = {3:MODEL_3G,4:MODEL_4G,5:MODEL_5G,6:MODEL_6G,7:MODEL_7G}


def getStatsFromOutput(output):
    pos = output.find('Perplexity = ')
    if pos == -1:
        return -1
    op = output[pos:]
    parts = op.split(',')
    perplexity = parts[0].split('=')
    perplexity = float(perplexity[1].strip())
    return perplexity    


def calcTextNGramPerplexity(dat,n=3):
    child = pexpect.spawn(EVAL_COMMAND.format(EVAL_SCRIPT,MODEL_PATH+NtoM[n]))
    #print EVAL_COMMAND.format(EVAL_SCRIPT,MODEL_PATH+NtoM[n])
    child.expect('eval',timeout=400)
    ret = [calculateNgramPerplexity("\n".join(dat[i]),child, n,i) for i in range(len(dat))]
    child.sendline('quit')
    child.close(force=True)
    child = None
    #print ret
    return ret
   

def calculateNgramPerplexity(txt,child,n=3,i=-1):
    with open(TEXT_FILE,'w') as f:
        f.write(txt)    
    child.sendline(PERP_COMMAND.format(TEXT_FILE))
    child.expect('eval')
    op = getStatsFromOutput(child.before)
    #print i
    return op

import string
import math
import random
import time

#normalizes input list to input scale. default scale is 1.
def normalize(l, scale = 1):
    tot = 0
    for i in l:
        tot += i
    if tot == 0:
        return l
    return [(x/tot)*scale for x in l]

def normalizeDict(d, scale = 1):
    tot = 0
    for i in d.values():
        tot += i
    if tot == 0:
        return d
    return { x:(d[x]/tot)*scale for x in d.keys()}    



def multinomial(sentence, data, vocab, Laplace = 0):
    #helper function for multinomial to calculate word probability.
    def multWordProb(word, data, vocabLength, Laplace = 0):
        freq = 0
        count = 0
        for sentence in data:
            count += len(sentence) #count keeps track of total number of words in class
            for words in sentence:
                if words == word:
                    freq += 1 #if appears in doc then freq++
        return  (freq + Laplace) / (count + Laplace * vocabLength)
    #end helper

    stTrans = str.maketrans('', '', string.punctuation) #to remove punctuation
    sentence = sentence.lower().translate(stTrans).split() #removes punctuation from input sentence



    corpusFreq = {x:len(data[x]) for x in data.keys()}
    corpusFreq = normalizeDict(corpusFreq)
    #initializes dict of probabilities
    prob = {k:0 for k in data.keys()}
    
    for k in prob.keys(): #calcs P(doc|class)
        for word in sentence:
            if word in vocab:
                prob[k] += math.log( multWordProb(word, data[k], len(vocab), Laplace)) #adding log is same as mult 
    total = 0
    for k in prob.keys():
        total += math.pow(math.e, prob[k] ) * corpusFreq[k] #sum of probs for denominator
    assert total > 0
    return {x:math.pow(math.e, prob[x])*corpusFreq[x]/total for x in prob.keys()} #calcs P(class|doc) using bayes rule


#calculates the probability of each word in the sentence existing in the given corpus. 
#run time is highly dependant on vocab size, num of classes, and len of corpus.
#since alg requires the whole vocab, the run time is subpar. 
def multBern(sentence, data, vocab, Laplace = 0):
    #multBern helper function that calculates the probability of the input word in the corupus
    #P(w_t| c_j; theta)
    def bernWordProb(word, data, numData, Laplace):
        freq = 0
        for s in data: #s is each review in data
            for w in s: #w is each word in review
                if w == word:
                    freq += 1 #if appears in doc then freq++
                    break #so it doesnt count how many times in doc, just bool if word is there or not
        #(num of docs word appears in + laplace) / (number of docs in data + num classes * laplace)
        # print((freq + Laplace) / (len(data) + numData * Laplace) )
        return (freq + Laplace) / (len(data) + numData * Laplace) 
    #end helper
    stTrans = str.maketrans('', '', string.punctuation) #to remove punctuation
    sentence = sentence.lower().translate(stTrans).split() #removes punctuation from input sentence

    #calculates probability of given corpus
    corpusFreq = {x:len(data[x]) for x in data.keys()}
    corpusFreq = normalizeDict(corpusFreq)
    #initializes dict of probabilities
    prob = {k:0 for k in data.keys()}
    count = 0
    for k in prob.keys(): #calcs P(doc|class)
        count += 1
        for v in vocab:
            p = bernWordProb(v, data[k], len(data), Laplace)
            if v in sentence:
                prob[k] += math.log(p)
            else:
                prob[k] += math.log(1-p)
        print(str(count) + ' completed')
    total = 0
    # for k in prob.keys():
    #     print(prob[k], corpusFreq[k])
    for k in prob.keys():
        total += math.pow(math.e, prob[k]) * corpusFreq[k] #sum of probs for denominator 
    assert total > 0
    
    return {x:math.pow(math.e, prob[x])*corpusFreq[x]/total for x in prob.keys()} #calcs P(class|doc) using bayes rule

def readRatingData():
    stTrans = str.maketrans('', '', string.punctuation) #to remove punctuation
    s = open(".\\ML\\drugLib_raw\\drugLibTrain_raw.tsv") #opens file
    infile = s.read().splitlines() #reads to list
    # infile = infile[1:int(len(infile)/2)]
    data={} #dict to store data in sorted manner
    for i in range(0, len(infile)-1): #splits based on tab
        infile[i] = infile[i].split('\t')
    for line in range(len(infile)-1, -1, -1): #removes incomplete lines
        if len(infile[line]) < 9:
            infile.pop(line)
    for x in range(1,11): #docs to appropriate classes in dict
        data[x] = [k[6].lower().translate(stTrans).split() + k[7].lower().translate(stTrans).split() + k[8].lower().translate(stTrans).split() for k in infile if k[2] == str(x)]
    return data

if __name__ == "__main__":
    data = readRatingData()
    vocab = {} #used to help process vocab
    setVocab = set() #final vocab set
    for k in data.keys():
        for x in data[k]:
            for word in x:
                #counts num of occurances of words
                if word in vocab.keys():
                    vocab[word] += 1
                else:
                    vocab[word] = 1
    #removes words with occurence <= 1
    #this improves run time and gets rid of uncommon words such as obscure drug names that have no effect of results
    for word in vocab.keys():
        if vocab[word] > 1:
            setVocab.add(word)
    stopWords = 'a an am and are as at be by because for from had has he her in is it its of on that the to was were will with or so'
    for word in stopWords.split():
        setVocab.remove(word)
    print('vocab done')
    # for c in data.keys():
    #     for s in data[c]:
    #         for i in range(len(s)-1, -1, -1):
    #             if s[i] not in vocab:
    #                 s.pop(i)
    total = 0
    startm = time.time()
    resultsm = multinomial("Decreased the vertigo almost immediately; no further vomiting; I was able to walk and move my head without the awful spinning.	Sedation--I almost the entire time I took the medication...I'm not sure I could have taken this and gone to work.	My doctor said the Labyrinthitis is usually preceded by a cold and fluid collects in the inner portion of the ear and causes the dizziness/vomiting.  The treatment was symptom relief (using the Meclizine) and the fluid gradually absorbs.", data, setVocab, 1)
    endm =  time.time() - startm
    startb = time.time()
    resultsb = multBern("Decreased the vertigo almost immediately; no further vomiting; I was able to walk and move my head without the awful spinning.	Sedation--I almost the entire time I took the medication...I'm not sure I could have taken this and gone to work.	My doctor said the Labyrinthitis is usually preceded by a cold and fluid collects in the inner portion of the ear and causes the dizziness/vomiting.  The treatment was symptom relief (using the Meclizine) and the fluid gradually absorbs.", data, setVocab, 1)
    endb = time.time() - startb
    print('Time: Bern: ' + str(endb) + ' Mult: ' + str(endm))
    print(resultsm)
    print(resultsb)
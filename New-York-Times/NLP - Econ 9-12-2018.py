# NLP
## Documentation
## https://www.nltk.org/api/nltk.tokenize.html

# Attached there is an example runnign nltk as the library to create tokens. please run the code in segments if using Spyder. There is an issue on line 18 with the IDE that causes running all the code at once to fail. 
# the documentation for the library is available on-line on the address provided.  

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('treebank')
nltk.download('vader_lexicon')
from nltk.corpus import treebank
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize

sent = """the dog chased the cat"""

sentences = ["Trade war is not funny.", # positive sentence example
...    "a fast growing world economy is awesome !", # punctuation emphasis handled correctly (sentiment intensity adjusted)
...    "a wide trade deficit is not a good thing to have in america today",  # booster words handled correctly (sentiment intensity adjusted)
...    "ECONOMIC COOPERATION is VERY SMART, handsome, and FUNNY.",  # emphasis for ALLCAPS handled
...    "GLOBAL ECONOMIC GROWTH is VERY SMART, handsome, and FUNNY!!!",# combination of signals - VADER appropriately adjusts intensity
...    "Unemployment is not good",# booster words & punctuation make this close to ceiling for score
...    "High Employment is incredibly FUNNY!!!",         # positive sentence
...    "The response from corporations was kind of good.", # qualified positive sentence is handled correctly (intensity adjusted)
...    "Costs to consumers can create a longer bussines cycle and that is not good", # mixed negation sentence
...    "A loss of confidence is A really bad, horrible thing",       # negative sentence with booster words
...    "At least it isn't a horrible book.", # negated negative sentence with contraction
...    ":) and :D",     # emoticons handled
...    "",              # an empty string is correctly handled
...    "Today sux",     #  negative slang handled
...    "Today sux!",    #  negative slang with punctuation emphasis handled
...    "Today SUX!",    #  negative slang with capitalization emphasis
...    "Today kinda sux! But I'll get by, lol" # mixed sentiment example with slang and constrastive conjunction "but"
... ]

para = "The world economy is in a deep recession. \
Two years have passed since America imposed tariffs on many of its trading partners, prompting retaliation from China, Canada, the European Union and others. \
Negotiations to resolve differences faltered amid tensions over trade surpluses and deficits. \
The effects of the protectionist measures seemed modest at first, when global economic growth was still fairly strong. \
But costs gradually started to add up for businesses and consumers. Investments faltered. Global supply chains choked. \
Then, in 2019, the American business cycle turned. \
In China, confidence in corporations’ ability to service debt fell. Financial markets plummeted. \
Surplus goods from China flooded into other markets, where pressure to raise import barriers became irresistible \
The downturn worsened. Job losses soared into the tens of millions. \
Pinging An boasts a booming health and life insurance businesses and technology first mindset with its focus on InsurTech and FinTech solutions. \
Yet its online lending affiliate Lufax, which became profitable last year, delayed a planned IPO this spring, and its Good Doctor IPO in Hong Kong in May fizzled. \
Still, industry insiders are watching for a planned IPO of its subsidiary One_Connect, which sells technology platforms to banks."

tricky_sentences = [
...    "Most automated sentiment analysis tools are shit.",
...    "VADER sentiment analysis is the shit.",
...    "Sentiment analysis has never been good.",
...    "Sentiment analysis with VADER has never been this good.",
...    "Warren Beatty has never been so entertaining.",
...    "I won't say that the movie is astounding and I wouldn't claim that \
...    the movie is too banal either.",
...    "I like to hate Michael Bay films, but I couldn't fault this one",
...    "It's one thing to watch an Uwe Boll film, but another thing entirely \
...    to pay for it",
...    "The movie was too good",
...    "This movie was actually neither that funny, nor super witty.",
...    "This movie doesn't care about cleverness, wit or any other kind of \
...    intelligent humor.",
...    "Those who find ugly meanings in beautiful things are corrupt without \
...    being charming.",
...    "There are slow and repetitive parts, BUT it has just enough spice to \
...    keep it interesting.",
...    "The script is not fantastic, but the acting is decent and the cinematography \
...    is EXCELLENT!",
...    "Roger Dodger is one of the most compelling variations on this theme.",
...    "Roger Dodger is one of the least compelling variations on this theme.",
...    "Roger Dodger is at least compelling as a variation on the theme.",
...    "they fall in love with the product",
...    "but then it breaks",
...    "usually around the time the 90 day warranty expires",
...    "the twin towers collapsed today",
...    "However, Mr. Carter solemnly argues, his client carried out the kidnapping \
...    under orders and in the ''least offensive way possible.''"
... ]
lines_list = tokenize.sent_tokenize(para)
sentences.extend(lines_list)
sentences.extend(tricky_sentences)
sid = SentimentIntensityAnalyzer()
for sentence in sentences:
    print(sentence)
    ss = sid.polarity_scores(sentence)
    for k in sorted(ss):
        print('{0}: {1}, '.format(k, ss[k]), end='')
    print()


tokens = nltk.word_tokenize(para)
tagged = nltk.pos_tag(tokens)
entities = nltk.chunk.ne_chunk(tagged)
#entities.pretty_print()
t = treebank.parsed_sents('wsj_0001.mrg')[0]
t.draw()

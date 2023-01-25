import random

acgt = ["a", "c", "g", "t"]
num = 10
length = 100

data_prefix = "compare_model/SPEID/sequence/GM12878/"
enhancers = data_prefix + "test_enhancer.fa"
promoters = data_prefix + "test_promoter.fa"

def make_random_seq():
    with open(enhancers, "w") as f:
        for _ in range(num*2):
            seq = ""
            if _ % 2 == 0:
                seq += f">test_enhancer{_}"
            else:
                for _ in range(length):
                    bp = acgt[random.randrange(4)]
                    seq += bp
            f.write(seq + "\n")

    with open(promoters, "w") as f:
        for _ in range(num*2):
            seq = ""
            if _ % 2 == 0:
                seq += f">test_promoter{_}"
            else:
                for _ in range(length):
                    bp = acgt[random.randrange(4)]
                    seq += bp
            f.write(seq + "\n")


make_random_seq()
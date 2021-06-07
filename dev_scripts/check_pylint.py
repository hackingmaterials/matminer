import os


source_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../pylint.txt")

with open(source_file, "r") as f:
    for l in f.readlines():
        if "Your code has been rated at" in l:
            toks = l.split(" ")
            score_tok = [t for t in toks if "/10" in t][0]
            score = float(score_tok.split("/")[0])
            if score < 9.5:
                raise ValueError(f"Pylint score is {score}, less than threshold of 9.5!")
            else:
                print("Pylint says code is ok")
                break
    else:
        raise ValueError("Pylint log file could not be parsed!")

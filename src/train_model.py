import argparse



if __name__=="__main__":
    parser=argparse.ArgumentParser(
        description="Setup dataset by downloading it from remote storage and extract it in a local subfolder")
    parser.add_argument("--dataset-dir","-dd",help="Root directory where the datasets have been extracted into", default="../dataset")
    
    
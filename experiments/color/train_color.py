import argparse
from framework.experiment import Command
from experiments.color.experiment import ColorExperiment

def run_experiment(args):
    allowed_color_list = [
        'black',
        'white',
        'red',
        'yellow',
        'blue',
        'gray'
    ]
    exp = ColorExperiment(class_names=allowed_color_list, **vars(args))
    if(Command.train_stats in args.commands):
        stats = exp.get_train_stats()
        print(stats)
    if(Command.test_stats in args.commands):
        stats = exp.get_test_stats()
        print(stats)
    if(Command.tune in args.commands):
        exp.tune_learning_rate()
    if(Command.train in args.commands):
        exp.train()
    if(Command.predict in args.commands):
        exp.predict_and_persist()
    if(Command.evaluate in args.commands):
        exp.evaluate_predictions()
    # prediction_path = exp.predict_and_persist()
    # exp.evaluate_predictions(prediction_path)
    
    


if __name__=="__main__":
    
    parser=argparse.ArgumentParser(
        description="Run experiments on color based models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser = ColorExperiment.add_parser_args(parser)
    
    args = parser.parse_args()
    
    run_experiment(args)
    
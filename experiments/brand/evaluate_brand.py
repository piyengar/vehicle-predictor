import argparse

from experiments.brand.experiment import BrandExperiment
from experiments.brand.dataset.brand import Brand

def run_experiment(args):
    allowed_brand_list = list(map(lambda x: x.name, [
        Brand.HYUNDAI,
        Brand.VOLKSWAGEN,
        Brand.BUICK,
        Brand.WULING,
        Brand.CHEVROLET,
        Brand.NISSAN,
        Brand.KIA,
        Brand.TOYOTA,
        Brand.AUDI,
        Brand.HONDA,
    ]))
    exp = BrandExperiment(class_names=allowed_brand_list, **vars(args))
    exp.predict_and_persist()
    exp.evaluate_predictions()
    

if __name__=="__main__":
    
    parser=argparse.ArgumentParser(
        description="Run experiments on brand based models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser = BrandExperiment.add_parser_args(parser)
    
    args = parser.parse_args()
    
    run_experiment(args)
    
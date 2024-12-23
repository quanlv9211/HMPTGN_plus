from script.utils.util import logger
from script.models.HMPTGN import HMPTGN
from script.models.HMPTGNplus.HMPTGNplus import HMPTGNplus


def load_model(args):
    if args.model == 'HMPTGN':
        model = HMPTGN(args)
    elif args.model == 'HMPTGNplus':
        model = HMPTGNplus(args)
    else:
        raise Exception('pls define the model')
    logger.info('using model {} '.format(args.model))
    return model

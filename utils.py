import torch


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def get_name(opts, exclude_keys=[], vals_only=False, no_datetime=False, ext='.pkl'):
    """
    Get a file-name based on a dictionary, set of dict keys to exclude from the name,
    whether to add the datetime string, and what extension, if any
    """
    name = None
    if not no_datetime:
        name = str(datetime.now()).replace(' ', '-').split('.')[0]
    for key in sorted(opts):
        if key not in exclude_keys:
            if vals_only:
                if name is None:
                    name = str(opts[key])
                else:
                    name = '_'.join((name, str(opts[key])))
            else:
                if name is None:
                    name = '-'.join((key, str(opts[key])))
                else:
                    name = '_'.join((name, '-'.join((key, str(opts[key])))))
    if ext is not None:
        name += ext
    return name

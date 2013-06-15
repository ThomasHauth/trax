import matplotlib as mpl

mpl.use('pgf')
latexConf = {
    "pgf.texsystem": "pdflatex",
    "font.family": "serif", # use serif/main font for text elements
    "axes.titlesize": "x-large",   # fontsize of the axes title
    "axes.labelsize": "xx-large",  # fontsize of the x any y labels
    "xtick.labelsize": "x-large",  # fontsize of the tick labels
    "ytick.labelsize": "x-large",  # fontsize of the tick labels
    "text.usetex": True,    # use inline math for ticks
    "pgf.rcfonts": False,   # don't setup fonts from rc parameters
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage{siunitx}",         # load additional packages
    ]}
mpl.rcParams.update(latexConf)
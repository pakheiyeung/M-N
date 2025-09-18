import os
import glob

from utilities import Results

current_dir = os.path.dirname(os.path.realpath(__file__))



'Display the results after you have run test.py'
files_list = glob.glob(os.path.join(current_dir, 'experiments', 'nih_6_pseudo', '*.pkl'))

df_dict = {}
for f in files_list:
    results = Results(results_path=f)
    print(f)
    df_final = results.print()

    df_dict.update({f: df_final})


'Save to a csv file '
if False:
    with open(os.path.join(current_dir, 'temp.csv'),'a') as f:
        for fname, df in df_dict.items():
            f.write(fname)
            f.write("\n")
            df.to_csv(f)
            f.write("\n")



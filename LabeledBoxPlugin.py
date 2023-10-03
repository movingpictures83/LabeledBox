"""
    Main Jasper script. BETA.
    Please do not use in production environments as this script is undergoing active development.
    If you would like to use Jasper in your research, please get in touch.
"""

import os, sys
import shutil
import argparse
import collections
import random
import time
import datetime
import math
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shlex
import subprocess as sp

#import  jasper_utilities as jutils


# ------------------------------------------------ Path to R Script ---------------------------------------------------
#
#   Get the path of the R script so that we can call it to draw the Hilbert curve.
#

class LabeledBoxPlugin:
  def input(self, inputfile):
      self.path_to_profile = inputfile
      self.profile_type = "matrix"
      self.order_scheme = "labeled"
      self.ignore_labels = "store_true"
      self.min_num_levels = 100
      self.verbose_output = "store_true"

  def run(self):
      pass

  def output(self, outputfile):
    self.output_path = outputfile
    print("[" + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + "]")
    print("[" + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + "] Starting Visualization...")
    print("[" + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + "] Ordering Scheme: " +
          str(self.order_scheme).upper())

    # -------------------------------------------- Output Directories -------------------------------------------------
    #
    #
    #print("[" + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + "] Preparing Output Directory...")
    #
    #output_dir = self.output_path + "/jasper-images"
    #
    #if not os.path.exists(output_dir):
    #    print("[" + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + "]" +
    #          "  - Output directory does not exist. Creating...")
    #    os.makedirs(output_dir)
    #
    #print("[" + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + "]")


    # -------------------------------------------- Temp Directories ---------------------------------------------------
    #
    #
    #temp_dir = output_dir + "/_tmp"
    #if not os.path.exists(temp_dir):
    #    os.makedirs(temp_dir)


    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------- Profile Type -----------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    #
    #   What type of profiles are we dealing with?
    #
    print("[" + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + "] Analyzing Profiles...")
    print("[" + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + "] Type: " + str(self.profile_type).upper())

    start_time = time.time()

    #
    #   'matrix' type contains profiles. Each 'row' is a sample, and it contains N-columns, with each column being
    #   a microbial strain, and the last column being a class label for the sample.
    #
    if self.profile_type == "matrix":

        if self.ignore_labels:
            print("[" + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + "]  - Ignoring labels in output.")

        print("[" + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + "] Loading...")

        #
        #   Profile's Data Frame. The main data structure in the program.
        #
        profile_df = pd.read_csv(self.path_to_profile, sep='\t', header=0, index_col=0, engine='python')

        feature_list   = list(profile_df.columns.values)
        sample_list    = list(profile_df.index.values)

        number_of_microbes = len(feature_list) - 1
        number_of_samples  = len(profile_df.index)
        loc_label = len(feature_list) - 1   # The column that contains the Sample's label.

        taxa_name_list = feature_list
        taxa_name_list.pop()

        print("[" + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + "] " + "Number of Microbes: " +
              '{:0,.0f}'.format(number_of_microbes))

        print("[" + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + "] " + "Number of Samples: " +
              '{:0,.0f}'.format(number_of_samples))

        print("[" + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + "]")

        print("[" + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + "] Creating Hilbert Images..." )


        # -------------------------------------------------------------------------------------------------------------
        #
        #   Ordering based on a Labeled Interpretation
        #
        if self.order_scheme == "labeled":
            print("[" + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + "] Reordering based on Labels...")

            #   Sort the dataframe based on the labels column
            df_sorted_by_labels = profile_df.sort_values("Label")

            #   New dataframe that will only have the average relative abundances of the cohorts
            df_mean_abundance = df_sorted_by_labels[0:0]

            #   Extract a unique list of the labels so that we can extract individual datasets.
            labels_list = list(set(list(df_sorted_by_labels["Label"])))
            labels_list.sort()
            print(labels_list)

            for label in labels_list:
                print("[" + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + "] Extracting " + str(label))

                #
                #   Extract the samples (rows) that have the same "Label" — or extract all the samples from a
                #   labeled condition identify by the label column.
                #
                cohort_df = df_sorted_by_labels[df_sorted_by_labels["Label"] == label]

                print("[" + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + "]   - Calculating mean...")

                cohort_df.loc["mean"] = cohort_df.mean()

                # Set the label of the last cell in the dataframe as the call to 'mean()' above leaves it blank.
                cohort_df.iloc[-1,-1] = label

                df_mean_abundance.loc[label] = cohort_df.loc["mean"]

            #   At this point, the dataframe "df_mean_abundance" should have as many rows as we have cohorts, and each
            #   row should contain the average relative abundance of the microbe in the cohort.

            #   We don't need the last column ('Label') anymore, so we discard it
            df_mean_abundance = df_mean_abundance.iloc[:, :-1]

            #   We'll transpose it so that we can easily sort it.
            df_mean_abundance_t = df_mean_abundance.transpose()
            df_col_names = list(df_mean_abundance_t.columns.values)
            df_mean_abundance_t = df_mean_abundance_t.sort_values(df_col_names, ascending=False)

            #   Boxplots
            boxplot = sns.boxplot(x="Sample Name",
                                  y="value",
                                  data=pd.melt(df_mean_abundance_t),
                                  palette="colorblind")

            boxplot.set_title("Raw Data")

            plot_file_path = self.output_path + ".boxplot.png"
            boxplot.figure.savefig(plot_file_path,
                                   format='png',
                                   dpi=120)


            #   Keep processing the Dataframe with more columns we need.
            df_mean_abundance_t["Max"] = df_mean_abundance_t[df_col_names].max(axis=1)
            print(df_mean_abundance_t)
            df_mean_abundance_t["Max Label"] = df_mean_abundance_t[df_col_names].idxmax(axis=1)

            sample_tmp_file = self.output_path+".abundance.tsv"

            df_mean_abundance_t.to_csv(sample_tmp_file,
                                       sep="\t",
                                       encoding="utf-8",
                                       header=True,
                                       index=True,
                                       line_terminator="\n")




    #
    #   We are done so we can calculate the overall runtime, display it, and finish.
    #
    end_time = time.time()
    run_time = end_time - start_time

    print("[" + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + "]")
    print("[" + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + "] " + "Analysis Run Time: " +
          str(datetime.timedelta(seconds=run_time)))
    print("[" + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + "] Done.")
    print("[" + time.strftime('%d-%b-%Y %H:%M:%S', time.localtime()) + "]")



# ---------------------------------------------------- End of Line ----------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

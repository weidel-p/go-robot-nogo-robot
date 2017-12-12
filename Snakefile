import os, glob

MODEL_DIR = 'code/striatal_model'
ANALYSIS_DIR = 'code/analysis'
DATA_PATH = 'data/'
THREADS = range(9) 

EXPERIMENTS = [os.path.basename(f) for f in glob.glob('code/striatal_model/experiments/*')]
SLIDING_EXPERIMENTS = ["no_stim.yaml", "bilateral_D1.yaml", "bilateral_D2.yaml", "sequencesd1d2.yaml"]
SEQ_MULT_EXPERIMENTS = ["sequencesMultTrials.yaml","sequencesMultTrialsd2.yaml"]

TRIALS = range(5)
SINGLE_TRIALS = range(1)

wildcard_constraints: 
    hemi="[a-z]+",
    trial="[0-9]",
    fn="^((?!-).)*",

rule all:
    input:
        "figs/competingActions.yaml/competing_traces_left.pdf",
        "figs/competingActions.yaml/competing_traces_right.pdf",
        "figs/competingActions.yaml/competing_corr_left.pdf",
        "figs/competingActions.yaml/competing_corr_right.pdf",

        "figs/competingActionsNoD2Conn.yaml/competing_traces_left.pdf",
        "figs/competingActionsNoD2Conn.yaml/competing_traces_right.pdf",
        "figs/competingActionsNoD2Conn.yaml/competing_corr_left.pdf",
        "figs/competingActionsNoD2Conn.yaml/competing_corr_right.pdf",

        'figs/competingActions.yaml/competing_corr_change_within_go_left_left.pdf',
        'figs/competingActions.yaml/competing_corr_change_within_go_right_right.pdf',
        'figs/competingActions.yaml/competing_corr_change_between_D1D1_left.pdf',
        'figs/competingActions.yaml/competing_corr_change_between_D2D2_right.pdf',

        expand('figs/{experiments}/corr_mean_activity_left.pdf', experiments=EXPERIMENTS),
        expand('figs/{experiments}/corr_mean_activity_right.pdf', experiments=EXPERIMENTS),

        expand('figs/{experiments}/corr_sliding_left.pdf', experiments=SLIDING_EXPERIMENTS),
        expand('figs/{experiments}/corr_sliding_right.pdf', experiments=SLIDING_EXPERIMENTS),

        expand("figs/{experiments}/new_corr_with_stim_left.pdf", experiments=EXPERIMENTS),
        expand("figs/{experiments}/new_corr_with_stim_right.pdf", experiments=EXPERIMENTS),
        expand("figs/{experiments}/new_corr_with_bckgrnd_left.pdf", experiments=EXPERIMENTS),
        expand("figs/{experiments}/new_corr_with_bckgrnd_right.pdf", experiments=EXPERIMENTS),

        expand("figs/{experiments}/corr_with_stim_left.pdf", experiments=EXPERIMENTS),
        expand("figs/{experiments}/corr_bw_stim_left.pdf", experiments=EXPERIMENTS),
        expand("figs/{experiments}/corr_with_stim_right.pdf", experiments=EXPERIMENTS),
        expand("figs/{experiments}/corr_bw_stim_right.pdf", experiments=EXPERIMENTS),
        expand("figs/{experiments}/corr_with_bckgrnd_left.pdf", experiments=EXPERIMENTS),
        expand("figs/{experiments}/corr_bw_bckgrnd_left.pdf", experiments=EXPERIMENTS),
        expand("figs/{experiments}/corr_with_bckgrnd_right.pdf", experiments=EXPERIMENTS),
        expand("figs/{experiments}/corr_bw_bckgrnd_right.pdf", experiments=EXPERIMENTS),

        expand("figs/{experiments}/corr_left.pdf", experiments=EXPERIMENTS),
        expand("figs/{experiments}/corr_right.pdf", experiments=EXPERIMENTS),

        expand("figs/{experiments}/trajectory.pdf", experiments=EXPERIMENTS),
        expand("figs/{experiments}/turnpoints.pdf", experiments=EXPERIMENTS),
        
        expand("figs/{experiments}/raster_plot_left_hemisphere.pdf", experiments=EXPERIMENTS),
        expand("figs/{experiments}/raster_plot_right_hemisphere.pdf", experiments=EXPERIMENTS),

        expand("figs/{experiments}/hist_left.pdf", experiments=EXPERIMENTS),
        expand("figs/{experiments}/hist_right.pdf", experiments=EXPERIMENTS),

        expand("figs/{experiments}/hist_left_3x3.pdf", experiments=EXPERIMENTS),
        expand("figs/{experiments}/hist_right_3x3.pdf", experiments=EXPERIMENTS),

        expand("figs/{experiments}/changeAct.pdf", experiments=SEQ_MULT_EXPERIMENTS),


rule clean:
    shell:
        """
        rm -r {}*
        """.format(DATA_PATH)

rule cleanFigs:
    shell:
        """
         echo Do you really have to delete \*all\* figures\? Think about it, it will take hours to run again.. \(y/N\);
         read decision;
         if [ "$decision" = "y" ]; then
             find . -name '*.pdf' -type f -delete
         else
             echo \"Good decision!\"
         fi
        """


rule plot_competing_change_CC:
    input:
        'data_long/competingActions.yaml/competing_corr_{hemi}.json',
        'data_long/competingActionsNoD2Conn.yaml/competing_corr_{hemi}.json',
    output:
        'figs/{experiment}/competing_corr_change_within_go_left_{hemi}.pdf',
        'figs/{experiment}/competing_corr_change_within_go_right_{hemi}.pdf',
        'figs/{experiment}/competing_corr_change_between_D1D1_{hemi}.pdf',
        'figs/{experiment}/competing_corr_change_between_D2D2_{hemi}.pdf',
    run:
        shell('ipython -c "%run code/analysis/plot_competing_correlations_change.py {input} {output}"')



rule plot_competing_CC:
    input:
        expand('data_long/{{experiment}}/{trial}/{{hemi}}_hemisphere.gdf', trial=TRIALS),
        expand('data_long/{{experiment}}/{trial}/neuron_ids_{{hemi}}_hemisphere.json', trial=TRIALS),
        'code/striatal_model/experiments/{experiment}',
    output:
        'figs/{experiment}/competing_traces_{hemi}.pdf',
        'figs/{experiment}/competing_corr_{hemi}.pdf',
        'data_long/{experiment}/competing_corr_{hemi}.json',
    run:
        shell('ipython -c "%run code/analysis/plot_competing_correlations.py {input} {wildcards.hemi} {output}"')



rule plot_sliding_CC:
    input:
        expand('data/{{experiment}}/{trial}/{{hemi}}_hemisphere.gdf', trial=TRIALS),
        expand('data/{{experiment}}/{trial}/neuron_ids_{{hemi}}_hemisphere.json', trial=TRIALS),
        'code/striatal_model/experiments/{experiment}',
    output:
        'figs/{experiment}/corr_sliding_{hemi}.pdf',
    run:
        shell('ipython -c "%run code/analysis/plot_sliding_correlations.py {input} {wildcards.hemi} {output}"')



rule plot_mean_act_CC:
    input:
        expand('data_long/{{experiment}}/{trial}/{{hemi}}_hemisphere.gdf', trial=SINGLE_TRIALS),
        expand('data_long/{{experiment}}/{trial}/neuron_ids_{{hemi}}_hemisphere.json', trial=SINGLE_TRIALS),
        'code/striatal_model/experiments/{experiment}',
    output:
        'figs/{experiment}/corr_mean_activity_{hemi}.pdf',
    run:
        shell('ipython -c "%run code/analysis/plot_mean_activity_correlations.py {input} {wildcards.hemi} {output}"')



rule plot_CC:
    input:
        expand('data_long/{{experiment}}/{trial}/{{hemi}}_hemisphere.gdf', trial=SINGLE_TRIALS),
        expand('data_long/{{experiment}}/{trial}/neuron_ids_{{hemi}}_hemisphere.json', trial=SINGLE_TRIALS),
        'code/striatal_model/experiments/{experiment}',
    output:
        'figs/{experiment}/corr_{hemi}.pdf',
    run:
        shell('ipython -c "%run code/analysis/plot_correlations.py {input} {wildcards.hemi} {output}"')


rule new_plot_CC_grid:
    threads: 5
    input:
        expand('data_long/{{experiment}}/{trial}/{{hemi}}_hemisphere.gdf', trial=SINGLE_TRIALS),
        expand('data_long/{{experiment}}/{trial}/neuron_ids_{{hemi}}_hemisphere.json', trial=SINGLE_TRIALS),
        'code/striatal_model/experiments/{experiment}',
    output:
        'figs/{experiment}/new_corr_with_stim_{hemi}.pdf',
        'figs/{experiment}/new_corr_with_bckgrnd_{hemi}.pdf',
    run:
        shell('ipython -c "%run code/analysis/plot_correlations_gridnew.py {input} {wildcards.hemi} {output}"')
        


rule plot_CC_grid:
    input:
        expand('data_long/{{experiment}}/{trial}/{{hemi}}_hemisphere.gdf', trial=SINGLE_TRIALS),
        expand('data_long/{{experiment}}/{trial}/neuron_ids_{{hemi}}_hemisphere.json', trial=SINGLE_TRIALS),
        'code/striatal_model/experiments/{experiment}',
    output:
        'figs/{experiment}/corr_with_stim_{hemi}.pdf',
        'figs/{experiment}/corr_bw_stim_{hemi}.pdf',
        'figs/{experiment}/corr_with_bckgrnd_{hemi}.pdf',
        'figs/{experiment}/corr_bw_bckgrnd_{hemi}.pdf',
    run:
        shell('ipython -c "%run code/analysis/plot_correlations_grid.py {input} {wildcards.hemi} {output}"')
        


rule plot_trajectories:
    input:
        expand('data/{{experiment}}/{trial}/odom.bag', trial=SINGLE_TRIALS),
        expand('data/{{experiment}}/{trial}/left_hemisphere.gdf', trial=SINGLE_TRIALS),
        expand('data/{{experiment}}/{trial}/right_hemisphere.gdf', trial=SINGLE_TRIALS),
        expand('data/{{experiment}}/{trial}/neuron_ids_left_hemisphere.json', trial=SINGLE_TRIALS),
        expand('data/{{experiment}}/{trial}/neuron_ids_right_hemisphere.json', trial=SINGLE_TRIALS),
        'code/striatal_model/experiments/{experiment}',
    output:
        'figs/{experiment}/trajectory.pdf',
        'figs/{experiment}/turnpoints.pdf',
    run:
        shell('ipython -c "%run code/analysis/trajectory_plotter.py {input} {output}"')

rule plot_activityChange_MultTrials:
    input:
        expand('data/{{experiment}}/{trial}/left_hemisphere.gdf', trial=SINGLE_TRIALS),
        expand('data/{{experiment}}/{trial}/right_hemisphere.gdf', trial=SINGLE_TRIALS),
        expand('data/{{experiment}}/{trial}/neuron_ids_left_hemisphere.json', trial=SINGLE_TRIALS),
        expand('data/{{experiment}}/{trial}/neuron_ids_right_hemisphere.json', trial=SINGLE_TRIALS),
        'code/striatal_model/experiments/{experiment}',
    output:
        'figs/{experiment}/changeAct.pdf',
    run:
        shell('ipython -c "%run code/analysis/plot_changeActivity_multTrials.py {input} {output}"')


rule concat_gdf:
    input:
        expand('{{fn}}-{threads}.gdf', threads=THREADS),
    output:
        '{fn}.gdf',
    run:
        shell('cat {wildcards.fn}-* > {wildcards.fn}.gdf')


rule plot_histogram:
    input:
        expand('data/{{experiment}}/{trial}/{{hemi}}_hemisphere.gdf', trial=SINGLE_TRIALS),
    output:
        'figs/{experiment}/hist_{hemi}.pdf',
    run:
        shell('ipython -c "%run code/analysis/plot_histogram.py {wildcards.experiment} {wildcards.hemi} {output}"')


rule plot_histogram3x3:
    input:
        expand('data/{{experiment}}/{trial}/{{hemi}}_hemisphere.gdf', trial=SINGLE_TRIALS),
    output:
        'figs/{experiment}/hist_{hemi}_3x3.pdf',
    run:
        shell('ipython -c "%run code/analysis/plot_histogram_3x3.py {wildcards.experiment} {wildcards.hemi} {output}"')
     

rule plot_raster:
    input:
        expand('data/{{experiment}}/{trial}/{{hemi}}_hemisphere.gdf', trial=SINGLE_TRIALS),
        expand('data_long/{{experiment}}/{trial}/neuron_ids_{{hemi}}_hemisphere.json', trial=SINGLE_TRIALS),
    output:
        'figs/{experiment}/raster_plot_{hemi}_hemisphere.pdf',
    run:
        shell('python code/analysis/plot_raster.py {input} {output}')
        

rule run_experiment:
    threads: 48
    output:
        'data/{experiment}/{trial}/odom.bag',
        expand('data/{{experiment}}/{{trial}}/left_hemisphere-{thread}.gdf', thread=THREADS),
        expand('data/{{experiment}}/{{trial}}/right_hemisphere-{thread}.gdf', thread=THREADS),
        'data/{experiment}/{trial}/neuron_ids_left_hemisphere.json',
        'data/{experiment}/{trial}/neuron_ids_right_hemisphere.json',
    params:
        'prefix = {experiment}',
        'scale = 1.',
        'trial = {trial}',
    run:
        shell('cd {dir}; python launch_trial.py {params}', dir=MODEL_DIR)


rule run_experiment_long:
    threads: 48
    output:
        'data_long/{experiment}/{trial}/odom.bag',
        expand('data_long/{{experiment}}/{{trial}}/left_hemisphere-{thread}.gdf', thread=THREADS),
        expand('data_long/{{experiment}}/{{trial}}/right_hemisphere-{thread}.gdf', thread=THREADS),
        'data_long/{experiment}/{trial}/neuron_ids_left_hemisphere.json',
        'data_long/{experiment}/{trial}/neuron_ids_right_hemisphere.json',
    params:
        'prefix = {experiment}',
        'scale = 10.',
        'trial = {trial}',
    run:
        shell('cd {dir}; python launch_trial.py {params}', dir=MODEL_DIR)
        


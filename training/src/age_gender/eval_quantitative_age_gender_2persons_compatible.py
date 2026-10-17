import os
import ast
import cv2
import logging
import datetime
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as A
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from src.utils.viewer_utils import get_face_bbox
from src.age_gender.inference import InferenceEngineAgeGender

def run_evaluation(input_path, model_path, output_path, config_file_path=None):
    pass

def plot_age_figures(eval_dir, output_dir):
    pass

def plot_gender_figures(eval_dir, output_dir):
    pass

def plot_detection_figures(eval_dir, output_dir):
    pass

def evaluate(input_path=None, model_path=None, output_path=None, already_computed_path=None, config_file_path=None):   
    
    if already_computed_path is None: 
        # Load config from yaml file if given
        if config_file_path is not None:
            from src.utils.config import Config
            from src.utils.data_augmentations import DataAugPipeline

            config = Config()
            config.set_config(config_file_path)

            # data_aug_pip = DataAugPipeline('eval',
            #          config['input']['data_augmentations']['test']['pipeline']['box_ops'],
            #          config['input']['data_augmentations']['test']['pipeline']['albumentation_ops'],
            #          config['input']['data_augmentations']['test']['pipeline']['custom_ops'],
            #          config['input']['data_augmentations']['test']['pipeline']['p'])


            albumentation_ops = config['input']['data_augmentations']['test']['pipeline']['albumentation_ops']
            p = config['input']['data_augmentations']['test']['pipeline']['p']

            print('albumentation_ops:\n', albumentation_ops)

            albumentation_pipeline = None
            list_data_aug = []
            for albumentation_op, params in albumentation_ops.items():
                if albumentation_op in A.__dict__.keys():
                    params2 = params.copy()
                    for key in params2.keys():
                        if type(params2[key])==str:
                            params2[key] = ast.literal_eval(params2[key])
                    list_data_aug.append(A.__dict__[albumentation_op](**params2))

            if len(list_data_aug) > 0:
                albumentation_pipeline = A.Compose(
                    list_data_aug,
                    p=p
                )


        else:
            print("No config file given, using no data augmentations.")
        

        # Create inference engine
        age_gender_inference = InferenceEngineAgeGender(model_path)
        logging.info(f"inference Engine for Age and gender successfully created.")

        # Get all files of input_path
        files_to_process = os.listdir(os.path.join(input_path, 'raw'))

        ######## TO REMOVE #####
        files_to_process = [
            'willy_kpedio_35_M_190_95.sec',
            'thibault_vincent_28_M_178_68.sec',
            'yang_yi_32_F_164_52.sec',
            'yves_lefouler_32_M_178_82.sec'
        ]
        ########################

        if 'test' in files_to_process:
            files_to_process.remove('test')
            print('removed test')

        # if 'Joseph_D_40_M.sec' in files_to_process:
        #     files_to_process.remove('Joseph_D_40_M.sec')
        #     print('removed Joseph_D_40_M.sec')

        if 'Victor_D_26_M.sec' in files_to_process:
            files_to_process.remove('Victor_D_26_M.sec')
            print('removed Victor_D_26_M.sec')


        logging.info(f"{len(files_to_process)} files to process.")

        dataset_dfs = []
        # Loop over files
        with tqdm(total=len(files_to_process), desc="Processing Files") as pbar_files:              

            for f in files_to_process:
                f_root = f.replace('.sec', '')
                logging.info(f"Processing {f_root}...")

                # Get data from file name
                gt_age_driver = int(f_root.split('_')[2])
                gt_gender_driver = f_root.split('_')[3]
                if len(f_root.split('_')) == 4:
                    print(f'\tFile with 1 person: {gt_age_driver}, {gt_gender_driver}')                    
                    gt_age_passenger = None
                    gt_gender_passenger = None
                elif len(f_root.split('_')) == 8:
                    gt_age_passenger = int(f_root.split('_')[6])
                    gt_gender_passenger = f_root.split('_')[7]
                    print(f'\tFile with 2 persons: {gt_age_driver}, {gt_gender_driver}, {gt_age_passenger}, {gt_gender_passenger}')
                else:
                    raise ValueError(f"File name {f} does not follow the expected format.")
                
                
                # Load video
                video_path = os.path.join(input_path, 'mpeg' , f_root + '.mp4')
                cap = cv2.VideoCapture(video_path)
                logging.info(f"\tVideo loaded, dimensions (LxHxW):" 
                            f"{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}x"
                            f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}x"
                            f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}")
                        
                # Load smarteye csv data
                csv_path = os.path.join(input_path, 'raw', f, 'log', 'tracking_000000.csv')
                df = pd.read_csv(csv_path)
                
                count = 0

                file_results = []

                with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc=f"Processing Frames of \"{f_root}\"") as pbar_frames:              
                    while(cap.isOpened()):
                        # read a frame
                        ret, frame = cap.read()
                        if ret:
                            # get crops and crops quality
                            head_crop0, head_crop_quality0, _, bbox_0 = get_face_bbox(df, occupant_id=0, frame=frame, frame_cnt=count)
                            head_crop1, head_crop_quality1, _, bbox_1 = get_face_bbox(df, occupant_id=1, frame=frame, frame_cnt=count)

                            if albumentation_pipeline:                            
                                # infer model on crops
                                pred0 = age_gender_inference([albumentation_pipeline(image=head_crop0)["image"]])[0] if head_crop0 is not None else None
                                pred1 = age_gender_inference([albumentation_pipeline(image=head_crop1)["image"]])[0] if head_crop1 is not None else None
                            else:
                                # infer model on crops
                                pred0 = age_gender_inference([head_crop0])[0] if head_crop0 is not None else None
                                pred1 = age_gender_inference([head_crop1])[0] if head_crop1 is not None else None


                            # Attribute occupant detections to driver or passenger positions
                            ##### WARNING, the following applies only for left driving cars! #####

                            frame_horizontal_middle = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 2)

                            print('pred0:', pred0)
                            print('pred1:', pred1)

                            driver_res = [None, None, None, None, None]
                            passenger_res = [None, None, None, None, None]

                            if pred0 is not None:
                                if bbox_0[0] <frame_horizontal_middle:
                                    driver_res = [{'M':0,'F':1}[pred0[0]], pred0[1], pred0[2], pred0[3], head_crop_quality0]
                                else:
                                    passenger_res = [{'M':0,'F':1}[pred0[0]], pred0[1], pred0[2], pred0[3], head_crop_quality0]
                            
                            if pred1 is not None:
                                if bbox_1[0] <frame_horizontal_middle:
                                    driver_res = [{'M':0,'F':1}[pred1[0]], pred1[1], pred1[2], pred1[3], head_crop_quality1]
                                else:
                                    passenger_res = [{'M':0,'F':1}[pred1[0]], pred1[1], pred1[2], pred1[3], head_crop_quality1]
                            print('driver_res:', driver_res)
                            print('passenger_res:', passenger_res)
                               

                            # add model results to list
                            if passenger_res[0] is not None and gt_age_passenger is None:
                                file_results.append([*(driver_res), *(passenger_res), None, gt_gender_driver, None, gt_age_driver, f_root, count])
                            else:                                
                                file_results.append([*(driver_res), *(passenger_res), gt_gender_driver, gt_gender_passenger, gt_age_driver, gt_age_passenger, f_root, count])
        

                            # update progress bar   
                            pbar_frames.update(1)
                            count += 1

                        else:
                            break
                
                df_file_results = pd.DataFrame(file_results,
                                                columns=[
                                                    'driver_gender', 'confidence_driver_gender',
                                                    'driver_age', 'confidence_driver_age',
                                                    'driver_head_crop_quality',
                                                    'passenger_gender', 'confidence_passenger_gender',
                                                    'passenger_age', 'confidence_passenger_age',
                                                    'passenger_head_crop_quality',
                                                    'GT_gender_driver', 'GT_gender_passenger',
                                                    'GT_age_driver', 'GT_age_passenger',
                                                    'file_name', 'frame_number'
                                                ])
                
                dataset_dfs.append(df_file_results)

                pbar_files.update(1)

        
        df_dataset_results = pd.concat(dataset_dfs)
        df_dataset_results.to_csv(os.path.join(output_path, 'results.csv'), index=False)

        # plot figures
        plot_figures(df_dataset_results, output_path)
        
    else:
        # Load already computed results
        csv_path = os.path.join(already_computed_path, 'results.csv')
        df_dataset_results = pd.read_csv(csv_path)
        logging.info(f"Results loaded from {csv_path}")

    # plot figures
        plot_figures(df_dataset_results, already_computed_path)


def plot_figures(df_results, output_dir):

    plot_detection_figures = False
    plot_age_figures = True
    plot_gender_figures = True

    # color-blind friendly colors
    colors = ['#E69F00', '#56B4E9', '#009E73', '#F0E442']  


    # Create directory to save figures
    figures_dir = os.path.join(output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    logging.info(f"Figures directory created: {figures_dir}")
    
    if plot_detection_figures:
        detection_figures_dir = os.path.join(figures_dir, 'detections')
        os.makedirs(detection_figures_dir, exist_ok=True)
        driver_detection_rates = []
        passenger_detection_rates = []
        filenames = []
        total_driver_n_detections = 0
        total_passenger_n_detections = 0



        for n, g in df_results.groupby('file_name'):
            print(n)
            f_driver_n_detections = len(g[~g.driver_gender.isna()])
            f_passenger_n_detections = len(g[~g.passenger_gender.isna()])
            print(f'\tDriver Detections: {f_driver_n_detections}')
            print(f'\tPassenger Detections: {f_passenger_n_detections}')

            filenames.append(n)
            total_driver_n_detections += f_driver_n_detections
            total_passenger_n_detections += f_passenger_n_detections

            detection_rate = (f_driver_n_detections + f_passenger_n_detections) / len(g) * 100
            driver_detection_rates.append(f_driver_n_detections/len(g)*100)
            passenger_detection_rates.append(f_passenger_n_detections/len(g)*100)

            print(f'\tDetection Rate:{np.round(detection_rate, 2)} %')
            
            driver_detections = np.where(g.driver_gender.isna(), 0, 1)
            passenger_detections = np.where(g.passenger_gender.isna(), 0, 1)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=g.reset_index().index, y=driver_detections, mode='lines',
                                     marker=dict(color=colors[1]),
                                     name="Driver Detections"))
            fig.add_trace(go.Scatter(x=g.reset_index().index, y=passenger_detections, mode='lines',
                                     marker=dict(color=colors[0]),
                                     name="Passenger Detections"))
            fig.update_layout(
                title=f"File name: {n} | Detection Rate:{np.round(detection_rate, 2)} %",
                xaxis_title="Frame number",
                yaxis_title="Number of Detections"
            )

            
            with open(os.path.join(detection_figures_dir, f'detections_{n}.html'), "w") as f:
                f.write(fig.to_html())
        
        # # Add data for global plot
        print(f'Global Driver Detections: {total_driver_n_detections}')
        print(f'Global Passenger Detections: {total_passenger_n_detections}')
        print(len(df_results))
        driver_detection_rates.append((total_driver_n_detections/len(df_results))*100)
        passenger_detection_rates.append((total_passenger_n_detections/len(df_results))*100)
        filenames.append('Global')

        # Plot dataset detection rates figure
    
        fig = go.Figure()

        driver_display_opacity = [0.5 for i in range(len(filenames)-1)]
        driver_display_opacity.append(1)
        driver_display_colors = [colors[0] for i in range(len(filenames))]        
        driver_display_dict = dict(color=driver_display_colors, opacity=driver_display_opacity)

        passenger_display_opacity = [0.5 for i in range(len(filenames)-1)]
        passenger_display_opacity.append(1)
        passenger_display_colors = [colors[1] for i in range(len(filenames))]        
        passenger_display_dict = dict(color=passenger_display_colors, opacity=passenger_display_opacity)


        fig.add_trace(go.Bar(x=filenames,
                            y=driver_detection_rates,
                            marker=driver_display_dict,
                            name='Driver',
                            offsetgroup=0
                            ))
        fig.add_trace(go.Bar(x=filenames,
                            y=passenger_detection_rates,
                            marker=passenger_display_dict,
                            name='Passenger',
                            offsetgroup=0,
                            base = driver_detection_rates  
                            ))
        
        
                                
        fig.update_layout(
            title=f"Percentage of frames with face detections| Dataset: recordings_pbys_23_01",
            xaxis_title="File Name",
            yaxis_title="Detection Rate (%)"
        )


        with open(os.path.join(figures_dir, f'detection_rates.html'), "w") as f:
            f.write(fig.to_html())

    if plot_age_figures:
        plot_figure_1 = True
        detection_figures_dir = os.path.join(figures_dir, 'age')
        os.makedirs(detection_figures_dir, exist_ok=True)

        if plot_figure_1:
            absolute_error_file = []
            filenames = []

            for n, g in df_results.groupby('file_name'):
                driver_valid_df = g[~g.driver_age.isna()]
                passenger_valid_df = g[~g.passenger_age.isna()]             

                # individual video plot of error through time
                if len(driver_valid_df) and len(passenger_valid_df):
                    min_index = np.min([driver_valid_df.index.values[0], passenger_valid_df.index.values[0]])
                    max_index = np.max([driver_valid_df.index[-1],  passenger_valid_df.index[-1]])
                elif not len(driver_valid_df):
                    min_index = passenger_valid_df.index.values[0]
                    max_index = passenger_valid_df.index[-1]
                else:
                    min_index = driver_valid_df.index.values[0]
                    max_index = driver_valid_df.index[-1]

                rolling_window_size = 5
                
                
                rollings = [20]
                
                for r in rollings:
                    passenger_valid_df[f'rolling_median_passenger_age_{r}'] = passenger_valid_df.passenger_age.rolling(window=r, min_periods=1).median()
                    driver_valid_df[f'rolling_median_driver_age_{r}'] = driver_valid_df.driver_age.rolling(window=r, min_periods=1).median()
                
                for _, row in driver_valid_df.iterrows():
                    absolute_error_file.append([n, 'driver', np.abs(row.driver_age - row.GT_age_driver)])
                    for r in rollings:
                        absolute_error_file.append([n, f'driver_{r}_rolling_median', np.abs(row[f'rolling_median_driver_age_{r}'] - row.GT_age_driver)])
                    
                for _, row in passenger_valid_df.iterrows():
                    absolute_error_file.append([n, 'passenger', np.abs(row.passenger_age - row.GT_age_passenger)])
                    for r in rollings:
                        absolute_error_file.append([n, f'passenger_{r}_rolling_median', np.abs(row[f'rolling_median_passenger_age_{r}'] - row.GT_age_passenger)])

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=driver_valid_df.index-min_index,
                                        y=driver_valid_df.driver_age, mode='markers', name='Driver Age Prediction'))
                
                # fig.add_trace(go.Scatter(x=driver_valid_df.index-min_index,
                #                         y=driver_valid_df.rolling_median_driver_age_5, mode='lines', name=f'5-median-rolling Driver Age Prediction'))
                
                fig.add_trace(go.Scatter(x=passenger_valid_df.index-min_index,
                                        y=passenger_valid_df.passenger_age, mode='markers', name='Passenger Age Prediction'))
                
                # fig.add_trace(go.Scatter(x=passenger_valid_df.index-min_index,
                #                         y=passenger_valid_df.rolling_median_passenger_age_10, mode='lines', name=f'10-median-rolling Passenger Age Prediction'))

                # fig.add_trace(go.Scatter(x=passenger_valid_df.index-min_index,
                #                         y=passenger_valid_df.rolling_median_passenger_age_15, mode='lines', name=f'15-median-rolling Passenger Age Prediction'))

                # Add cumulative mean of driver age
                driver_valid_df['cumulative_mean_driver_age'] = driver_valid_df.driver_age.expanding().median()
                fig.add_trace(go.Scatter(x=driver_valid_df.index-min_index,
                                        y=driver_valid_df.cumulative_mean_driver_age, mode='lines', name='Cumulative Mean Driver Age'))

                # Add cumulative mean of passenger age
                passenger_valid_df['cumulative_mean_passenger_age'] = passenger_valid_df.passenger_age.expanding().mean()
                fig.add_trace(go.Scatter(x=passenger_valid_df.index-min_index,
                                        y=passenger_valid_df.cumulative_mean_passenger_age, mode='lines', name='Cumulative Mean Passenger Age'))


                if len(driver_valid_df):
                    fig.add_trace(go.Scatter(x=driver_valid_df.index-min_index,
                                        y=[driver_valid_df.GT_age_driver.values[0] for _ in range(max_index-min_index)],
                                        mode='lines',
                                        name='Ground Truth Driver'))
                    
                if len(passenger_valid_df):
                    fig.add_trace(go.Scatter(x=passenger_valid_df.index-min_index,
                                            y=[passenger_valid_df.GT_age_passenger.values[0] for _ in range(max_index-min_index)],
                                            mode='lines',
                                            name='Ground Truth Passenger'))         
                
                                        
                with open(os.path.join(detection_figures_dir, f'{n}.html'), "w") as f:
                    f.write(fig.to_html())


            df = pd.DataFrame(absolute_error_file, columns=['file_name', 'position', 'absolute_error'])
            df['set'] = 'individual'
            df_global = df.copy(deep=True)
            df_global['file_name'] = 'Dataset Overall'
            df_global['set'] = 'Overall'

            df = pd.concat([df, df_global], axis=0)

            
            fig = px.box(df, x='file_name', y='absolute_error', color='position')#,
                            # color_discrete_sequence=colors)#, facet_col='set')            
            fig.update_yaxes(matches=None)
            fig.update_traces(boxmean=True)

            index_global_result = len(df_results.file_name.unique()) +0.5
            fig.add_vrect(x0=index_global_result-1, x1=index_global_result +0.1, line_width=0, fillcolor="grey", opacity=0.2)
            
            fig.update_layout(
                title=f"Box plots of age prediction absolute error | Dataset: recordings_pbys_23_01",
                xaxis_title="File Name",
                yaxis_title="Age Absolute Error (years)",
            )

            with open(os.path.join(figures_dir, f'age_boxplot.html'), "w") as f:
                f.write(fig.to_html())

        ####################
        # Plot scatter of predictions
        df_results_valid = df_results[~df_results.driver_age.isna()]
        fig = go.Figure()            
        fig.add_trace(go.Scatter(x=df_results_valid.GT_age_driver, y=df_results_valid.driver_age, mode='markers'))
        fig.update_layout(
            title='Scatter of age predictions with respect to ground truths for dataset: recordings_pbys_23_01',
            # title=f"File name: {n} | Age Absolute Error (mean, std):{np.round(np.mean(absolute_error_file[-1]), 2)}, "
            #     f"{np.round(np.std(absolute_error_file[-1]), 2)} years",
            xaxis_title="Ground Truth (years)",
            yaxis_title="Frame Predictions (years)"
        )
        with open(os.path.join(figures_dir, f'age_error_scatter.html'), "w") as f:
            f.write(fig.to_html())

        ####################

        # Plot boxplot of predictions w.r.t. age GT
        df_results_valid = df_results[(~df_results.driver_age.isna()) |
                                        (~df_results.passenger_age.isna())]
        df_results_valid['error_age'] = np.abs(df_results_valid.driver_age - df_results_valid.GT_age_driver)
        fig = px.box(df_results_valid, y='error_age', x='GT_age_driver')

        fig.update_traces(boxmean=True)

        fig.update_layout(
            title='boxplots of absolute age error with respect to ground truth age for dataset: recordings_pbys_23_01',
            # title=f"File name: {n} | Age Absolute Error (mean, std):{np.round(np.mean(absolute_error_file[-1]), 2)}, "
            #     f"{np.round(np.std(absolute_error_file[-1]), 2)} years",
            xaxis_title="Ground Truth Age (years)",
            yaxis_title="Absolute Prediction Error (years)"
        )

        with open(os.path.join(figures_dir, f'age_bloxplot_by_age.html'), "w") as f:
            f.write(fig.to_html())

        ####################

        # Plot boxplot of predictions w.r.t. gender
        df_results_valid = df_results[(~df_results.driver_age.isna()) |
                                        (~df_results.passenger_age.isna())]
        df_results_valid['error_age'] = np.abs(df_results_valid.driver_age - df_results_valid.GT_age_driver)
        
        df_results_valid['gender'] = df_results_valid['driver_gender'].combine_first(df_results_valid['passenger_gender'])
        df_results_valid['gender'] = df_results_valid['gender'].apply(lambda x: 'M' if int(x) == 0 else 'F')

        fig = px.box(df_results_valid, y='error_age', x='gender', points="all")
        fig.update_traces(boxmean=True, boxpoints=False)

        fig.update_layout(
            title='Boxplots of absolute age error with respect to ground truth gender for dataset: recordings_pbys_23_01',
            # title=f"File name: {n} | Age Absolute Error (mean, std):{np.round(np.mean(absolute_error_file[-1]), 2)}, "
            #     f"{np.round(np.std(absolute_error_file[-1]), 2)} years",
            xaxis_title="Ground Truth Gender (years)",
            yaxis_title="Absolute Prediction Error (years)"
        )

        with open(os.path.join(figures_dir, f'age_bloxplot_by_gender.html'), "w") as f:
            f.write(fig.to_html())

        ####################

        # Plot boxplot of age median on S first seconds
        S = 10
        data = []
        for s in np.arange(0.1,15,0.1):
            s_frames = int(s *23)


            df_results_valid = df_results[(~df_results.driver_age.isna()) |
                                            (~df_results.passenger_age.isna())]
            df_results_valid = df_results_valid.loc[df_results_valid.frame_number < s_frames]


            df_results_valid['error_age'] = np.abs(df_results_valid.driver_age - df_results_valid.GT_age_driver)
            
            df_results_valid['gender'] = df_results_valid['driver_gender'].combine_first(df_results_valid['passenger_gender'])
            df_results_valid['gender'] = df_results_valid['gender'].apply(lambda x: 'M' if int(x) == 0 else 'F')

            df_results_valid['s'] = s
            data.append(df_results_valid)

            
            median_error_by_file = df_results_valid.groupby('file_name')['error_age'].mean()
            # print(median_error_by_file)
            print(s, np.mean(median_error_by_file))
        
        data = pd.concat(data)

        fig = px.box(data, y='error_age', x='s')
        fig.update_traces(boxmean=True)
        fig.update_layout(
            title='Boxplots of absolute age error with respect to number of seconds into recording: recordings_pbys_23_01',
            # title=f"File name: {n} | Age Absolute Error (mean, std):{np.round(np.mean(absolute_error_file[-1]), 2)}, "
            #     f"{np.round(np.std(absolute_error_file[-1]), 2)} years",
            xaxis_title="Time from beginning of recording (seconds)",
            yaxis_title="Absolute Prediction Error (years)"
        )
        with open(os.path.join(figures_dir, f'S_first_frames_age.html'), "w") as f:
            f.write(fig.to_html())


        # fig = go.Figure()
        # fig.add_trace(go.Scatter(x=df_results_valid.groupby('file_name').GT_age.head(1), 
        #                          y=median_error_by_file,
        #                         mode='markers'))
        

        # fig.update_layout(
        #     title='Boxplots of absolute age error with respect to ground truth gender for dataset: recordings_pbys_23_01',
        #     # title=f"File name: {n} | Age Absolute Error (mean, std):{np.round(np.mean(absolute_error_file[-1]), 2)}, "
        #     #     f"{np.round(np.std(absolute_error_file[-1]), 2)} years",
        #     xaxis_title="Ground Truth Gender (years)",
        #     yaxis_title="Absolute Prediction Error (years)"
        # )

        # with open(os.path.join(figures_dir, f'median_of_{S}_first_frames_age.html'), "w") as f:
        #     f.write(fig.to_html())
        

    if plot_gender_figures:

        valid_driver = df_results[~df_results.driver_gender.isna()]
        valid_passenger = df_results[~df_results.passenger_gender.isna()]

        valid_driver['driver_gender'] = valid_driver['driver_gender'].apply(lambda x: 'M' if int(x) == 0 else 'F')
        valid_passenger['passenger_gender'] = valid_passenger['passenger_gender'].apply(lambda x: 'M' if int(x) == 0 else 'F')

        # Compute confusion matrix using pandas
        conf_matrix = pd.crosstab(valid_driver['GT_gender_driver'], valid_driver['driver_gender'],
                                rownames=['Actual'], colnames=['Predicted'],
                                normalize=0)

        # Display the confusion matrix
        print(conf_matrix)

        classes = ['M', 'F']
        inv_classes = ['F', 'M']

        fig = make_subplots(rows=1, cols=2, 
                            subplot_titles= ['Driver', 'Passenger']
                            )

        fig.add_trace(go.Heatmap(
            z=conf_matrix,
            x=classes,  # Predicted labels
            y=inv_classes,  # True labels
            colorscale='Blues',
            showscale=True),
            row = 1,
            col=1)
        
        fig.update_layout(
            title=f"Gender Confusion Matrix | Dataset: recordings_pbys_23_01",
        )


        conf_matrix = pd.crosstab(valid_passenger['GT_gender_driver'], valid_passenger['passenger_gender'],
                                rownames=['Actual'], colnames=['Predicted'],
                                normalize=0)
        print(type(conf_matrix))

        fig.add_trace(go.Heatmap(
            z=conf_matrix,
            x=classes,  # Predicted labels
            y=inv_classes,  # True labels
            colorscale='Blues',
            showscale=False),
            row = 1,
            col=2)

        fig.update_xaxes(title_text="Predicted Gender", row=1, col=1)
        fig.update_xaxes(title_text="Predicted Gender", row=1, col=2)
        fig.update_yaxes(title_text="True Gender", row=1, col=1)
        fig.update_yaxes(title_text="True Gender", row=1, col=2)
              

        with open(os.path.join(figures_dir, f'gender_confidence_matrix.html'), "w") as f:
            f.write(fig.to_html())
                 


    
    



    
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser("Age and gender quantitative evaluation")
    parser.add_argument(
        '-i',"--input_path", required=False, help="Path to input data folder"
     )
    parser.add_argument(
        '-m',"--model_path", required=False, help="Path to input model"
     )
    parser.add_argument(
        '-o',"--out_path", required=False, help="Path to store output results"
    )
    parser.add_argument(
        '-c',"--config_file_path", required=False, help="Path to config file path"
    )
    parser.add_argument(
        '-n',"--eval_name", required=False, help="Name for the folder to be created"
    )
    parser.add_argument(
        '-a',"--already_computed_path", required=False, help="Path to the directory that contains a csv file with already computed results. To be used when only plotting figures."
    )
    args = parser.parse_args()

    
    # Run evaluation
    if args.already_computed_path is None:
        logging.info(f"Starting complete evaluation...")

        # Create unique output directory 
        date_str = datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")
        output_dir = os.path.join(args.out_path, args.eval_name, date_str)
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Output directory created: {output_dir}")

        evaluate(
            input_path=args.input_path,
            model_path=args.model_path,
            output_path=output_dir,
            config_file_path=args.config_file_path
        )
    else:
        logging.info(f"Starting only figures plot...")
        evaluate(
            already_computed_path=args.already_computed_path
        )

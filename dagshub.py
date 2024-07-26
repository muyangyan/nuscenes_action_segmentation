import dagshub
import mlflow

dagshub.init("nuscenes_action_segmentation", "mu", mlflow=True)
mlflow.start_run()


mlflow.log_param
mlflow.set_tracking_uri('https://dagshub.com/<DagsHub-user-name>/<repository-name>.mlflow')

# Degugging gears with fw-beta
https://flywheel-io.gitlab.io/tools/app/cli/fw-beta/login/

### Once you have the Dockerfile and manifest.json, upload the gear to your FW instance
- The manifest gear-builder 'image' must match the 'version' and 'docker tag'
- (FW_API_KEY is an environment variable with your API key): 
- if running in vscode might need to source    
   
> source ~/.zshrc 
  
> fw-beta login --api-key=${FW_CLI_API_KEY} .   

### In the folder where you have your gear (Dockerfile and manifest), run:
> fw-beta gear build 

building flywheel container from contents of current directory
> fw-beta gear upload .

upload gear to flywheel instance (requires being signed in)

- On the instance, check installed gears on left hand side. There should be the description provided.
- Now, go to your instance and create a job run selecting the desired inputs and config options.
- Once you launch it, in the left side of the screen, click on "Jobs Log", 
- localize the job you just generated and click on it. Click on the "Log tab".
- At the top of the log, it will say: "Job Log for <my_job_id>"
- (<my_job_id> is the unique job ID, and will change from job to job).
- Copy the <my_job_id>.
* This may take a while to start running, but we now have the config options saved as we need and can instsantly downlaod the job from the command line. This gives us the full Flywheel structure which will help for debugging. 

### Then, run:
> fw-beta job pull <my_job_id> . 

**requires site admin privlidges** 
- fw-beta will download the gear Docker image if not present, plus the gear inputs, config, etc., in a folder called <my_gear_name>-<my_gear_version>-<my_job_id>
- Add your API key to the config file by running:

> cd <my_gear_name>-<my_gear_version>-<my_job_id>

> fw-beta gear config -i api-key=$FW_CLI_API_KEY

> cd ..

# API key is not downloaded in the config file from the instance. Now that it has been added to the config local debugging can take place. It is important not to share!! Only to run this locally for debugging. 

Now you are ready to `docker run` and debug/test the gear locally.
You just need to mount the same folders into your docker container as
are specified in the <my_gear_name>-<my_gear_version>-<my_job_id>/run.sh
(the lines that start with "-v ...")

If you want to do the debugging on PyCharm/VS Code, you need to use the 
Python interpreter inside your Docker image, and mount the same folders
specified in "run.sh".

- run.sh contains the docker image in flywheel, the docker commands including the mounted information
- The config contains all the options used in running the gear including those selected by the user. 

### run this container interactivly 
If entrypoint has been defined in Dockerfile, this can be overcome interactivly by:
docker run -it --entrypoint /bin/bash <image>

 `--rm`  good practice to remove container, rather than it spinning in the background when shut down 

 `-v` mount points (access to files/data)

 `\` allows continuation on return
 Can manually specify entry point to bash if different in Dockerfile, so as to easily navigate

> docker run -it --rm -v ""\
        flywheel/ciso:0.0.1-beta

The structure of this container is how the Flywheel instance sees it so now can debug easier by checking the input folder and config.json to make sure they match. There will be sample data there now from Flywheel that was specified by the user on the instance. 

- Any references now can be ammended
- The algorithm can now be updated with the required inputs 

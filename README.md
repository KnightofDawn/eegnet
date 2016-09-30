Repository of a neural network approach to detect epileptic seizures 
in iEEG data for the kaggle Melbourne Univeristy challenge.

After cloning the repository, run the following docker command:
docker run -p 6006:6006 -p 8888:8888 -it --name=eegnet --volume=YOUR_GITREPO_PATH:/shared projectappia/dqn:latest-cpu

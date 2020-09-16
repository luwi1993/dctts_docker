# dctts_docker
docker setup to start dctts implementation by Kyubyong https://github.com/Kyubyong/dc_tts

step 0:
    build docker image from Dockerfile with tag dctts (install docker)

step 1: 
    fix source path in docker_run.sh

step 2:
    download pretrained model https://github.com/Kyubyong/dc_tts
    
step 3:
    create your own sentences in text_input directory 
    
step 4: 
    run docker_run.sh 
    
step 5: 
    see results in the directory you linked to /dctts/samples/ in step 1
    
step 6:
    world domination! (please dont!) 
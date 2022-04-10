This will build docker image and name it 'math-ia':
  docker build -t math-ia .

  Explanation: 
    docker:     the command that starts almost everything docker does
    build:      build a new image from a Dockerfile that contains all the steps necessary
                A docker 'image' is a template for creating containers which are like virtual machines
    -t math-ia: tag/name the image 'math-ia'
    .:          the Dockerfile (and any the supporting it needs) are in the current directory

This will run the 'math-ia' image, with a network connection for your browser:
  docker run --rm -it -p 6080:6080 --name math-ia math-ia

  Explanation
    docker:         The command that starts almost everything docker does
    run:            Start a new container (kind of like a lightweight virtual machine)
    --rm:           Remove the container when it stops
    -it:            The container should be interactive and receive input from the current terminal
    -p 6080:6080    All programs (like your web browser) to connect to the network of the container on port 6080
    --name math-ia: Name the container 'math-ia' which helps find it later if you want
    math-ia:        The new container should be based on image math-ia (that was built above)

You can hit control-c in the docker-run window to kill the container. You can also run 'docker rm -f math-ia' 
in a different window to remove it if control-c is not working.
    


# Use Remote AITom Service via Visual Studio Code

## Install AITom on your Server

The server should be Linux based.

## Install SSH Extensions on Visual Studio Code

- This extension strongly recommends using key-based authentication (if you use a username/password, you'll be prompted to enter your credentials more than once by the extension)[[*](https://code.visualstudio.com/docs/remote/ssh-tutorial)];

- It does not support X11 Forwarding when using username/password authentication.

- Before connecting to your server, you may want to generate your own SSH key pairs and add your key to the server.

### Windows 10

- Search and install the following extensions:
  - [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh)  
    - Extension ID:  `ms-vscode-remote.remote-ssh` 
  - [Remote - SSH: Editing Configuration Files](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh-edit) 
    - Extension ID:  `ms-vscode-remote.remote-ssh-edit` 
- Please also check out the **Remote - SSH** tutorial: [Remote development over SSH](https://code.visualstudio.com/docs/remote/ssh-tutorial) 
- Make sure you enable OpenSSH service on your windows machine.

### Linux Distributions: Ubuntu 20.04

* Install the same extensions on your VS Code: [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) and [Remote - SSH: Editing Configuration Files](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh-edit).

* Microsoft's official SSH tutorial: [Remote development over SSH](https://code.visualstudio.com/docs/remote/ssh-tutorial) covers how to get started with these extensions. A more detailed tutorial about setting up SSH key-based authentication is available here: [Configuring SSH Key-based Authentication on Ubuntu 20.04](https://www.answertopia.com/ubuntu/configuring-ssh-key-based-authentication-on-ubuntu/).

* First, install and start the SSH service: 

  ```
  # apt install openssh-server
  # systemctl start sshd.service
  # systemctl enable sshd.service
  ```

* Then generate a key pair with command *ssh-keygen* and copy the public key onto the remote server with *ssh-copy-id*. (Note: this is for general setup. DM server admin with your public key and let him/her set it up for you.)

* Your key-based authentication should be set up at this point, follow Microsoft's extension tutorial to connect to the remote server.

### MacOS X

+ Install the same extensions on your VS Code: [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) and [Remote - SSH: Editing Configuration Files](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh-edit).
+ Please also check out the **Remote - SSH** tutorial: [Remote development over SSH](https://code.visualstudio.com/docs/remote/ssh-tutorial) 
+ You can generate a key pair the same way above. 

## Install X Server Client

Make sure your remote server has X server running.

### Windows 10

- Install any Windows X Server e.g. [Xming](https://sourceforge.net/projects/xming/), [Cygwin](https://x.cygwin.com/), we use and recommend [VcXsrv Windows X Server](https://sourceforge.net/projects/vcxsrv/).

### Linux Distributions: Ubuntu 20.04

* Ubuntu and most other linux distros have X services installed, so no extra step required. 

### MacOS X

+ X11 is no longer shipped with the Mac, so we have to install the [XQuartz](https://www.xquartz.org) project which will provide X11 server and client libraries.

## Configure Remote - SSH X11 Forwarding

- Edit your `settings.json`, put the following configuration to the head of the settings. This makes your VS Code terminal capable of forwarding the X11 window to your local client.

  ```json
  {
  "terminal.integrated.env.windows": {
          "DISPLAY":"localhost:0.0"
      },
      ... ...
  }
  ```

### <a name="startxsrv"></a> Start Windows X Server

- Start your [VcXsrv Windows X Server](https://sourceforge.net/projects/vcxsrv/).
  - Select your preferred display settings, click *Next*.
  - Select start no client, click *Next*.
  - Click *Next* and click *Finish*, you should be all set.

### <a name="startXorgandaddxhost"></a> Start Linux X Service and Allowing Remote Client's Access

* With X service already running, simply type the following command:

  ```
  $ xhost + <address of the server>
  ```

### <a name="startxquartz"></a> Start MacOS X Server

+ Start your [XQuartz](https://www.xquartz.org).
  + Give the server access to your local X Server.
  
    ```
    % xhost + server_ip
    ```

## Configure and Connect to your Host Server

1. Follow the instructions on the **Remote - SSH** tutorial: [Remote development over SSH](https://code.visualstudio.com/docs/remote/ssh-tutorial) 
2. Tips:
   - When you bring up the list of Remote extension commands, remember to put -X or -Y parameters in order to connect to the remote X server. e.g. `ssh -Y -p 22 aitom@scs.cmu.edu` 
   - Please also check your SSH `config` file, make sure it has `ForwardX11 yes` or `ForwardX11Trusted yes` options enabled.
3. In VS Code terminal, after connected to the host, enter `xclock` command to validate if you set up all the steps correctly.

## Configure Remote Python Interpreter

- To set up your remote python interpreter, please refer to [Using Python environments in VS Code](https://code.visualstudio.com/docs/python/environments).

## Workflow

### Windows 10

1. [Start Windows X Server](#startxsrv) 
2. Open VS Code and connect to your host.

### Linux Distributions: Ubuntu

1. No hassle, connect to your host in VS Code and enjoy!

### MacOS

1. [Start XQuartz](#startxquartz).
2. Open VS Code and connect to your host.

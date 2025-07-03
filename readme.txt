PortrART: Drawing Portraits Made Easy

Description: 

This project revolves around creating a digital art software, made by a portrait artist, for portrait artists. The software is complete with 
features found in any professional digital art software like different size brushes, brush colors, flood fill, eraser, color dropper, etc. In addition, the software offers unique features which have never been previously implemented. This includes getting the different positions of the facial features of the reference image and giving them to the user so that it becomes easy for them to maintain proportions while drawing. It also includes checking how accurate the user's drawing is, as well as being able to freely save the drawing and start from any base drawing that the user want. The user is able to upload their reference and base images. 

This software makes drawing portraits a lot easier, especially for beginner artists, and offers a user-friendly, easy to use UI.

Run Instructions:

In order to run the program, please download the entire zip file as it is.

Step 1: Unzip the folder
Step 2: Open the folder named 'src' and 'portrArt.py'.
Step 3: Open portrArt.py in a python interpreter, preferably VSCode. You will have to install cmu_graphics, PIL, cv2, os and math libraries if you do not already have them.
Step 4: After installing the libraries, run the code.

Optional Steps

We have provided 10 sample images of portraits you can use to serve as reference images by default. These are called 'Example1', 'Example2', ..., till 'Example10'. In order to upload them as reference images, click 'upload' after running the program, type in the name of the image you want (no need to add .jpeg or any extension). Once typed out, hit 'Done'. You can similarly also upload them as base images and start drawing on them.

On the other hand, you can access your last saved work also in the folder '15112_Face_Detection_Images', under the name 'userDrawing'. If you want to use it as a reference image or a base image to draw on, you can follow a similar process as mentioned above, by just typing 'userDrawing' after clicking on 'upload' in the drawing program.

Lastly, if you want to use your own reference images, you need to add them to the file '15112_Face_Detection_Images' in .jpeg format (ensure it is .jpeg). Additionally, note that it must contain a front facing face in it, or the program will not find any face an reject the image. To use it as a base or reference, simply type its name in 'upload' in the program after putting it in the correct folder.

All Program Features/ Shortcut Commands

In order to activate anything, simply click on it.

Top Left - 	Color Bar, allows user to choose any color for the brush or flood fill by clicking on it
Middle Left - 	Brush Selection, allows user to choose size 7, 5 or 3 brush and different shapes for the brush
Bottom Left -	Portrait Tools, for drawing portraits more easily
			- Portrait Guide, adds guide lines to the reference image and canvas according to user's choice
				- Image, adds/ removes all guiding tools from reference image
				- Boxes, adds/ removes boxes around facial features
				- Lines, adds/ removes all important lines on the face
				- Dots, adds/ removes all important points of interest on the face
			- Test Accuracy
				- Computes and returns the accuracy of all facial features, even tells if no face is found.
			
Top Row -	Tool Bar, contains various tools
			- Brush, standard brush tools; can make strokes
			- Eraser, erases anything on the canvas
			- Flood Fill, fills in all the surrounding space of the same color
			- Dropper, can pick up any color from the canvas or the reference image
			- Upload, used to upload reference and base images; once the names are typed out, hit 'Done' to implement changes
			- Save, saves the users drawing in '15112_Face_Detection_Images', under the name 'userDrawing'

For a quick tutorial, see 'video-demo.txt'

Happy portrArting fellow artists!

			

import javafx.application.Application;
import javafx.stage.Stage;
import javafx.scene.Scene;
import javafx.scene.layout.HBox;
import java.io.File;
import javafx.scene.media;
import javafx.scene.media.MediaView;
import javafx.scene.media.MediaPlayer;

public class VideoDemo extends Application{

    public static void main(String[] args){
        launch(args);
    }

    @Override
    public void start(Stage primaryStage){
	final double WIDTH = 640.0, HEIGHT = 490.0;
        File f = new File("Sample.mp4");
        Media media = new Media (videoFile.toURI().toString());

	MediaPlayer player = new MediaPlayer(media);
	player.setAutoPlay(true);

	MediaView view= new MediaView(player);

	view.setFitWidth(WIDTH);
	view.setFitHeight(HEIGHT);

	HBox hbox=new HBox(view);
	
	Scene scene= new Scene(hbox);
	primarySatge.setScene(scene);

	primaryStage.show();
	}
}



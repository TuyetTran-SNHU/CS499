import java.awt.*;
import javax.swing.*;


public class HomePage2
{

    public static void main (String[]args) 
    { 

        JFrame f1 = new JFrame ();
        f1.setSize (900, 1070);


        GridBagLayout gb = new GridBagLayout ();
        f1.setLayout (gb);



        GridBagConstraints gc = new GridBagConstraints ();
        JLabel label = new JLabel();
        label.setIcon(new ImageIcon("logo3.png"));

        TextField tf = new TextField (90);
	tf.setText("Search");

        Button b3 = new Button ("Enter");
        gc.fill = GridBagConstraints.HORIZONTAL;
        gc.weightx = 0.5;
        gc.weighty = 0.5;
        gc.gridx = 0;
        gc.gridy = 0;
        f1.add (label, gc);

        gc.gridx = 1;
        gc.gridy = 0;
        f1.add (tf, gc);

        gc.gridx = 2;
        gc.gridy = 0;
        f1.add (b3, gc);

        Button bb = new Button("Home");
	bb.setSize(1,1);
        gc.gridx = 0;
        gc.gridy = 1;
        f1.add (bb,gc);

	JLabel label1 = new JLabel("Hello!");
 	JLabel label2 = new JLabel("Hi welcome to Brandona.");
	JLabel label3 = new JLabel("We are a green clothing site.");
	JLabel label4 = new JLabel("This website is mainly focus in your needs.");
	JLabel label5 = new JLabel("You can browse through ready to wear clothes,");
	JLabel label6 = new JLabel("tailor your exisiting garment or customize a ");
	JLabel label7 = new JLabel("brand new piece.");
	JLabel label8 = new JLabel("Thank you.");
	JLabel label9 = new JLabel("Brandona.");
	JLabel label10 = new JLabel("To continue please enter click the service page.");

        gc.gridx = 0;
        gc.gridy = 2;
        f1.add (label1, gc);
        gc.gridx = 0;
        gc.gridy = 4;
        f1.add (label2, gc);
        gc.gridx = 0;
        gc.gridy = 6;
        f1.add (label3, gc);
        gc.gridx = 0;
        gc.gridy = 8;
        f1.add (label4, gc);
        gc.gridx = 0;
        gc.gridy = 10;
        f1.add (label5, gc);
        gc.gridx = 0;
        gc.gridy = 12;
        f1.add (label6, gc);
        gc.gridx = 0;
        gc.gridy = 14;
        f1.add (label7, gc);
	label8.setFont(new Font ("Allura", Font.BOLD,12));
        gc.gridx = 0;
        gc.gridy = 16;
        f1.add (label8, gc);
        gc.gridx = 0;
        gc.gridy = 18;
        f1.add (label9, gc);
        gc.gridx = 0;
        gc.gridy = 20;
        f1.add (label10, gc);

        Button b4 = new Button ("Button 4");
        gc.gridx = 0;
        gc.gridy = 0;
        gc.gridwidth = 3;
        gc.ipady = 10;


        Button b5 = new Button ("Service Page");
        gc.gridx = 2;
        gc.gridy = 20;
        gc.insets = new Insets (10, 0, 10, 0);
        f1.add (b5, gc);

        f1.pack ();
        f1.setVisible (true);
    }
}
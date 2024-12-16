import java.awt.*;
import javax.swing.*;
public class Service
{
    JFrame f;
    Service ()
    {
        f = new JFrame ();
        JLabel l1 = new JLabel ("Enter Name");
        JTextField tf1 = new JTextField (10);
        JButton b1 = new JButton ("SUBMIT");
        f.add (l1);
        f.add (tf1);
        f.add (b1);
        f.setLayout (new FlowLayout (FlowLayout.RIGHT));
        
        //setting flow layout of right alignment  
        f.setSize (300, 300);
        f.setVisible (true);
    }
    public static void main (String[]args)
    {
        new Service ();
    }
}
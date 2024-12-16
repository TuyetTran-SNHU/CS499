import java.awt.*;
public class BorderLayoutDemo
{
    public static void main (String[]args)
    {
        Frame f1 = new Frame ();
        f1.setSize (250, 250);
        Button b1 = new Button ("Button1");
        Button b2 = new Button ("Button2");
        Button b3 = new Button ("Button3");
        Button b4 = new Button ("Button4");
        Button b5 = new Button ("Button5");
        f1.add (b1, BorderLayout.NORTH);
        f1.add (b2, BorderLayout.EAST);
        f1.add (b3, BorderLayout.WEST);
        f1.add (b4, BorderLayout.SOUTH);
        f1.add (b5);
        f1.setVisible (true);
    }
}
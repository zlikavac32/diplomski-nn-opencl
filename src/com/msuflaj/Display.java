package com.msuflaj;

import com.martiansoftware.jsap.FlaggedOption;
import com.martiansoftware.jsap.JSAP;
import com.martiansoftware.jsap.JSAPResult;
import com.msuflaj.encoder.Encoder;
import com.msuflaj.encoder.EquilateralEncoder;
import com.msuflaj.network.FullyConnectedForwardNetwork;
import com.msuflaj.network.Network;
import com.msuflaj.network.NetworkException;
import com.msuflaj.storage.FileWieghtsStorage;
import com.msuflaj.storage.WeightsStorage;
import com.msuflaj.transfer.TanHFunction;
import com.msuflaj.transfer.TransferFunction;

import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.event.MouseInputListener;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.*;

public class Display {

    private static class Canvas extends JComponent implements MouseInputListener {

        private final Encoder encoder;

        private final String[] classes;

        private final Network network;

        private final String sampler;

        private final int gridN;

        private final int n;

        private final JLabel label;

        private java.util.List<Integer> x;

        private java.util.List<Integer> y;

        private java.util.List<Integer> t;

        private long startTime;

        private long lastTime;

        public Canvas(Network network, String[] classes, Encoder encoder, String sampler, int n, int gridN, JLabel label) {
            x = new ArrayList<>();
            y = new ArrayList<>();
            t = new ArrayList<>();
            this.network = network;
            this.classes = classes;
            this.encoder = encoder;
            this.sampler = sampler;
            this.n = n;
            this.gridN = gridN;
            this.label = label;
        }

        @Override
        public void paintComponent(Graphics g) {

            Graphics2D g2d = (Graphics2D) g;

            g2d.setColor(Color.BLACK);
            g2d.setStroke(new BasicStroke(2));
            for (int i = 1, s = x.size(); i < s; i++) {
                g2d.drawLine(x.get(i - 1), y.get(i - 1), x.get(i), y.get(i));
            }
        }

        @Override
        public void mouseDragged(MouseEvent e) {
            long currentTime = System.currentTimeMillis();

            if (currentTime - lastTime < 20) {
                return ;
            }

            lastTime = currentTime;

            x.add(e.getX());
            y.add(e.getY());
            t.add((int) (currentTime - startTime));

            repaint();
        }

        @Override
        public void mouseMoved(MouseEvent e) { /* ignore */ }

        @Override
        public void mouseClicked(MouseEvent e) { /* ignore */ }

        @Override
        public void mousePressed(MouseEvent e) {
            x.clear();
            y.clear();
            t.clear();
            startTime = System.currentTimeMillis();
            lastTime = startTime;

            x.add(e.getX());
            y.add(e.getY());
            t.add(0);
        }

        @Override
        public void mouseReleased(MouseEvent e) {

            if (x.size() < 3) { return; }

            java.util.List<Double> newX = new ArrayList<>();
            java.util.List<Double> newY = new ArrayList<>();

            newX.add((double) x.get(0));
            newY.add((double) y.get(0));

            double step = t.get(t.size() - 1) / (n - 1.);
            double temp = step;

            for (int i = 2; i < n; i++, temp += step) {
                int index = binarySearch(t, temp);
                newX.add(interpolate(x, temp, t, index));
                newY.add(interpolate(y, temp, t, index));
            }

            newX.add((double) x.get(x.size() - 1));
            newY.add((double) y.get(y.size() - 1));

            double[] data ;

            switch (sampler) {
                case "sin_cos" :
                    data = generateDataSinCos(newX, newY);
                    break;
                default :
                    data = generateDataGrid(newX, newY);
                    break;
            }

            double[] ret;

            try {
                ret = network.process(data);
            } catch (NetworkException e1) {
                e1.printStackTrace();
                return ;
            }

            int index = encoder.decode(ret);

            label.setText(classes[index]);

        }

        @Override
        public void mouseEntered(MouseEvent e) { /* ignore */ }

        @Override
        public void mouseExited(MouseEvent e) { /* ignore */ }

        private double[] generateDataGrid(java.util.List<Double> x, java.util.List<Double> y) {
            double minX = Collections.min(x);
            double minY = Collections.min(y);
            double rangeX = Collections.max(x) - minX;
            double rangeY = Collections.max(y) - minY;

            double[] data = new double[x.size()];

            for (int i = 0, c = x.size(); i < c; i++) {
                data[i] = ((int) ((x.get(i) - minX) / rangeX * (gridN - 1)) * gridN +
                        (int) ((y.get(i) - minY) / rangeY * (gridN - 1))) / (double) (gridN * gridN);
            }

            return data;
        }

        private double[] generateDataSinCos(java.util.List<Double> x, java.util.List<Double> y) {

            double[] data = new double[(x.size() - 1) * 2];

            for (int i = 1; i < x.size(); i++) {
                double tempX = x.get(i) - x.get(i - 1);
                double tempY = y.get(i) - y.get(i - 1);

                double len = Math.sqrt(tempX * tempX + tempY * tempY);

                if (len < 1e-9) {
                    data[(i - 1) << 1] = 0;
                    data[((i - 1) << 1) + 1] = 1;
                    continue;
                }

                data[(i - 1) << 1] = tempY / len;
                data[((i - 1) << 1) + 1] = tempX / len;

            }

            return data;
        }

        private int binarySearch(java.util.List<Integer> data, double value) {
            int l = 0;
            int h = data.size() - 1;
            while (l < h) {
                int m = (l + h) / 2;
                if (data.get(m) <= value) {
                    l = m + 1;
                } else {
                    h = m;
                }
            }

            return l;
        }

        private double interpolate(java.util.List<Integer> values, double v, java.util.List<Integer> time, int index) {
            return Math.abs(values.get(index) - values.get(index - 1)) / (double) (time.get(index) - time.get(index - 1)) *
                    (v - time.get(index - 1)) + values.get(index - 1);
        }

    }

    public static void main(String[] args) throws Exception {

        JSAP parser = new JSAP();

        FlaggedOption opt = new FlaggedOption("config")
                .setStringParser(JSAP.STRING_PARSER)
                .setDefault("conf.properties").setShortFlag('c')
                .setLongFlag("config");

        parser.registerParameter(opt);

        JSAPResult config = parser.parse(args);

        if (!config.success()) {
            printUsage(parser);
        }

        File configFile = new File(config.getString("config"));

        Properties props = new Properties();
        props.load(new BufferedReader(new FileReader(configFile)));

        String dimensions = props.getProperty("network.dimensions");

        if (null == dimensions) {
            error("network.dimensions");
            return ;
        }

        String[] parts = dimensions.split("x|X");

        int[] dimensionsConverted = new int[parts.length];

        if (parts.length < 2) {
            error("There need to be at least input and output layer", false);
            return;
        }

        for (int i = 0; i < parts.length; i++) {
            try {
                dimensionsConverted[i] = Integer.parseInt(parts[i]);
                if (dimensionsConverted[i] < 1) {
                    error("Layer " + (i + 1) + " has neuron count less than 0", false);
                    return;
                }
            } catch (NumberFormatException e) {
                error("network.dimensions");
                return ;
            }
        }

        TransferFunction f = new TanHFunction();

        TransferFunction[] functions = new TransferFunction[dimensionsConverted.length - 1];

        for (int i = 0; i < functions.length; i++) {
            functions[i] = f;
        }

        final Network network = new FullyConnectedForwardNetwork(dimensionsConverted, functions, true);

        String weightsFile = props.getProperty("network.weights");

        if (null == weightsFile) {
            error("network.weights");
            return;
        }

        File weights = new File(weightsFile);

        WeightsStorage storage = new FileWieghtsStorage(weights);
        if (!storage.load(network)) {
            error("Unable to load weights", false);
            return ;
        }

        String samplerString = props.getProperty("display.sampler");

        if (null == samplerString) {
            error("display.sampler");
            return;
        }

        final String sampler = samplerString.toLowerCase();

        if (
            !"sin_cos".equals(sampler)
            &&
            !"grid".equals(sampler)
        ) {
            error("display.sampler");
            return;
        }

        String displayClasses = props.getProperty("display.classes");

        if (null == displayClasses) {
            error("display.classes");
            return;
        }

        final String[] classes = new TreeSet<>(Arrays.asList(
            displayClasses.split(",")
        )).toArray(new String[1]);

        final Encoder encoder = new EquilateralEncoder(classes.length);

        int temp;

        try {
            temp = Integer.parseInt(props.getProperty("display.sampler.n"));
        } catch (NumberFormatException e) {
            error("display.sampler.n");
            return;
        }

        if (temp < 1) {
            error("display.sampler.n");
            return;
        }

        final int n = temp;

        try {
            temp = Integer.parseInt(props.getProperty("display.sampler.grid.n"));
        } catch (NumberFormatException e) {
            error("display.sampler.grid.n");
            return;
        }

        if (temp < 1) {
            error("display.sampler.grid.n");
            return;
        }

        final int gridN = temp;

        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {

                JPanel gui = new JPanel();
                gui.setLayout(new BorderLayout());

                JLabel label = new JLabel(" ");
                label.setBorder(BorderFactory.createCompoundBorder(
                    BorderFactory.createMatteBorder(1, 0, 0, 0, Color.BLACK),
                    new EmptyBorder(5, 2, 5, 2)
                ));

                Canvas c = new Canvas(network, classes, encoder, sampler, n, gridN, label);
                gui.add(c, BorderLayout.CENTER);
                gui.add(label, BorderLayout.SOUTH);
                gui.addMouseListener(c);
                gui.addMouseMotionListener(c);

                JFrame frame = new JFrame();
                frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
                frame.setTitle("Demonstracijski program");
                frame.setSize(500, 500);
                frame.add(gui);
                frame.setVisible(true);
            }
        });

    }

    private static void error(String error, boolean useKey) {
        if (useKey) {
            System.err.println("Invalid value for config key " + error);
        } else {
            System.err.println(error);
        }
        System.exit(2);
    }

    private static void error(String error) {
        error(error, true);
    }

    private static void printUsage(JSAP parser) {
        System.err.println();
        System.err.println("Usage: java -cp:/lib/ " + Train.class.getName());
        System.err.println("                " + parser.getUsage());
        System.err.println();
        System.err.println(parser.getHelp());
        System.exit(1);
    }

}

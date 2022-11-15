import java.util.stream.DoubleStream;

public class DevelopmentTesting {

    public static void main(String[] args) {

        double[][] a = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        DoubleStream.of(a[0]).sum();
    }
}

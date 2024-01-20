
import com.flag4j.*;
import com.flag4j.core.dense.DenseTensorBase;
import com.flag4j.io.PrintOptions;
import com.flag4j.rng.RandomTensor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TempTest {

    /**
     * Finds differences between two dense tensors and prints them out.
     * @param a First tensor to compare.
     * @param b Second tensor to compare.
     */
    public static <T extends DenseTensorBase<?, ?, ?, ?, ?>> List<int[]> findDiff(T a, T b) {
        if(!a.shape.equals(b.shape)) {
            System.out.printf("Not the same shape: %s and %s\n", a.shape, b.shape);
        }

        int stop = a.totalEntries().intValueExact();
        List<int[]> diffIndices = new ArrayList<>();

        for(int i=0; i<stop; i++) {
            int[] indices = a.shape.getIndices(i);

            if(!a.get(indices).equals(b.get(indices))) {
                System.out.printf("Difference at %s: %s, %s\n",
                        Arrays.toString(a.shape.getIndices(i)),
                        a.get(indices),
                        b.get(indices));
                diffIndices.add(indices);
            }
        }

        return diffIndices;
    }


    public static void main(String[] args) {
        PrintOptions.setMaxRowsCols(100);
        PrintOptions.setPrecision(100);
        Shape shape = new Shape(4096, 4096);
        double sparsity = 0.9999;
        RandomTensor rtg = new RandomTensor();

        int numRuns = 1;
        long start;
        double t1 = 0;
        double t2 = 0;

        for(int i=0; i<numRuns; i++) {
            CooMatrix coo = rtg.randomCooMatrix(shape, 0, 10, sparsity);
            CsrMatrix A = coo.toCsr();
            Matrix ADense = A.toDense();

            Matrix B = rtg.randomMatrix(shape.dims[1], shape.dims[0], 0, 10);

            start = System.nanoTime();
            Matrix c1 = A.mult(B);
            t1 += (System.nanoTime()-start)*1.0e-6;

            start = System.nanoTime();
            Matrix c2 = ADense.mult(B);
            t2 += (System.nanoTime()-start)*1.0e-6;

            findDiff(c1, c2);
        }

        t1 /= numRuns;
        t2 /= numRuns;

        System.out.printf("Shape1=%s, Shape2=%s, numRuns=%d\n", shape, shape.copy().swapAxes(0, 1), numRuns);
        System.out.printf("sparse-dense Time: %.3f ms\n", t1);
        System.out.printf("dense-dense Time: %.3f ms\n", t2);
    }
}

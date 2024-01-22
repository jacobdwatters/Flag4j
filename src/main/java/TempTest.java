import com.flag4j.CsrMatrix;
import com.flag4j.Matrix;
import com.flag4j.PermutationMatrix;
import com.flag4j.Shape;
import com.flag4j.core.dense.DenseTensorBase;
import com.flag4j.io.PrintOptions;
import com.flag4j.rng.RandomArray;
import com.flag4j.rng.RandomTensor;
import com.flag4j.util.ArrayUtils;

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


    public static void multiRuin() {
        boolean debug = true;
        Shape shape = new Shape(10, 10);
        RandomArray rag = new RandomArray();
        RandomTensor rtg = new RandomTensor();

        int numRuns = 1;
        long start;
        double t1 = 0;
        double t2 = 0;

        for(int i=0; i<numRuns; i++) {
            Matrix A = rtg.randomMatrix(shape, -10, 10);
            PermutationMatrix P = new PermutationMatrix(rag.shuffle(ArrayUtils.intRange(0, A.numCols)));
            Matrix pDense = P.toDense();

            start = System.nanoTime();
            Matrix c1 = P.rightMult(A);
            t1 += (System.nanoTime()-start)*1.0e-6;

            start = System.nanoTime();
            Matrix c2 = A.mult(pDense);
            t2 += (System.nanoTime()-start)*1.0e-6;

            if(debug) findDiff(c1, c2);
        }

        t1 /= numRuns;
        t2 /= numRuns;

        System.out.printf("Shape1=%s, numRuns=%d\n", shape, numRuns);
        System.out.printf("Specialized Permutation Time: %.3f ms\n", t1);
        System.out.printf("Explicit Dense Multiplication Time: %.3f ms\n", t2);
    }


    public static void main(String[] args) {
        PrintOptions.setMaxRowsCols(100);
        PrintOptions.setPrecision(100);

        double[][] uEntries = {
                {1, 3, 0, 0},
                {0, 0, 5, 0},
                {0, 0, 0, 4},
                {0, 0, 0, -91.3},};

        double[][] lEntries = {
                {1, 0, 0, 0},
                {0, 0, 0, 0},
                {0, 5.1, 0, 0},
                {0, 13.4, 0, -91.3}};

        CsrMatrix L = new Matrix(lEntries).toCsr();
        CsrMatrix U = new Matrix(uEntries).toCsr();
        U = U.set(-1, 2, 0);

        System.out.println(L + "\n");
        System.out.println(U + "\n" + "-".repeat(50) + "\n");

        System.out.println(L.getRow(0).toDense());
        System.out.println(L.getRow(1).toDense());
        System.out.println(L.getRow(2).toDense());
        System.out.println(L.getRow(3).toDense() + "\n\n");

        System.out.println(L.getCol(0).toDense());
        System.out.println(L.getCol(1).toDense());
        System.out.println(L.getCol(2).toDense());
        System.out.println(L.getCol(3).toDense() + "\n\n");
    }
}

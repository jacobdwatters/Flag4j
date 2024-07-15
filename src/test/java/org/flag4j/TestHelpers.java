package org.flag4j;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.core.MatrixMixin;
import org.flag4j.core.dense_base.DenseTensorBase;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TestHelpers {

    public static void printAsNumpyArray(Object... args) {
        for(Object arg : args) {
            if(arg instanceof MatrixMixin) {
                printAsNumpyArray((MatrixMixin<?, ?, ?, ?, ?, ?, ?>) arg);
            } else {
                System.out.print(arg.toString());
            }
        }
    }


    public static void printAsJavaArray(Object... args) {
        for(Object arg : args) {
            if(arg instanceof MatrixMixin) {
                printAsJavaArray((MatrixMixin<?, ?, ?, ?, ?, ?, ?>) arg);
            } else {
                System.out.print(arg.toString());
            }
        }
    }


    private static <T extends MatrixMixin<?, ?, ?, ?, ?, ?, ?>> void printAsNumpyArray(T A) {
        System.out.println(" = np.array([");

        for(int i=0; i<A.numRows(); i++) {
            System.out.print("\t[");
            for(int j=0; j<A.numCols(); j++) {

                if(A instanceof CMatrix) {
                    CMatrix B = (CMatrix) A;
                    if(B.get(i, j).im > 0) {
                        System.out.print(B.get(i, j).re + "+" + B.get(i, j).im + "j");
                    } else {
                        System.out.print(B.get(i, j).re + B.get(i, j).im + "j");
                    }
                } else {
                    // Then must be real.
                    System.out.print(A.get(i, j));
                }

                if(j < A.numCols()-1) {
                    System.out.print(", ");
                }
            }
            System.out.print("]");

            if(i < A.numRows()-1) {
                System.out.println(",");
            }
        }

        System.out.println("\n])");
    }


    private static <T extends MatrixMixin<?, ?, ?, ?, ?, ?, ?>> void printAsJavaArray(T A) {
        System.out.println("{");

        for(int i=0; i<A.numRows(); i++) {
            System.out.print("\t{");
            for(int j=0; j<A.numCols(); j++) {
                if(A instanceof CMatrix) {
                    CMatrix B = (CMatrix) A;
                    System.out.print("new CNumber(" + B.get(i, j).re + ", " + B.get(i, j).im + ")");
                } else {
                    // Then must be real.
                    System.out.print(A.get(i, j));
                }

                if(j < A.numCols()-1) {
                    System.out.print(", ");
                }
            }
            System.out.print("}");

            if(i < A.numRows()-1) {
                System.out.println(",");
            }
        }

        System.out.println("\n};");
    }


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
}

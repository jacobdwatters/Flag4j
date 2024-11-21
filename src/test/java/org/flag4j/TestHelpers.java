package org.flag4j;

import org.flag4j.arrays.backend.DenseTensorMixin;
import org.flag4j.arrays.backend.MatrixMixinOld;
import org.flag4j.arrays.backend.VectorMixinOld;
import org.flag4j.arrays.backend_new.AbstractTensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class TestHelpers {

    public static void printAsNumpyArray(Object... args) {
        for(Object arg : args) {
            if(arg instanceof MatrixMixinOld<?,?,?,?,?,?>) {
                printAsNumpyArray((MatrixMixinOld<?,?,?,?,?,?>) arg);
            } else {
                System.out.print(arg.toString());
            }
        }
    }


    public static void printAsJavaArray(Object... args) {
        for(Object arg : args) {
            if(arg instanceof MatrixMixinOld<?,?,?,?,?,?>) {
                printAsJavaArray((MatrixMixinOld<?,?,?,?,?,?>) arg);
            } else if(arg instanceof VectorMixinOld<?,?,?,?,?>) {
                printAsJavaArray((VectorMixinOld<?,?,?,?,?>) arg);
            } else {
                System.out.print(arg.toString()); // Type not found, fall back to toString method.
            }
        }
    }


    private static <T extends MatrixMixinOld<?,?,?,?,?,?>> void printAsNumpyArray(T A) {
        System.out.println(" = np.array([");

        for(int i=0; i<A.numRows(); i++) {
            System.out.print("\t[");
            for(int j=0; j<A.numCols(); j++) {

                if(A instanceof CMatrix B) {
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

            if(i < A.numRows()-1) System.out.println(",");
        }

        System.out.println("\n])");
    }


    private static <T extends VectorMixinOld<?,?,?,?,?>> void printAsJavaArray(T A) {
        System.out.print("{");

        for(int i=0; i<A.length(); i++) {
            if(A instanceof CVector B) {
                System.out.print("new Complex128(" + B.get(i).re + ", " + B.get(i).im + ")");
            } else {
                // Then must be real.
                System.out.print(A.get(i));
            }

            if(i < A.length()-1) System.out.print(", ");
        }
        System.out.println("};");
    }


    private static <T extends MatrixMixinOld<?,?,?,?,?,?>> void printAsJavaArray(T A) {
        System.out.println("{");

        for(int i=0; i<A.numRows(); i++) {
            System.out.print("\t{");
            for(int j=0; j<A.numCols(); j++) {
                if(A instanceof CMatrix B) {
                    System.out.print("new Complex128(" + B.get(i, j).re + ", " + B.get(i, j).im + ")");
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
    public static <T extends AbstractTensor<?,?,?>> List<int[]> findDiff(T a, T b) {
        if(!(a instanceof DenseTensorMixin<?,?> && b instanceof DenseTensorMixin<?,?>)) {
            System.out.println("Tensors are not dense.");
        }
        if(!a.shape.equals(b.shape)) {
            System.out.printf("Not the same shape: %s and %s\n", a.shape, b.shape);
        }

        int stop = a.totalEntries().intValueExact();
        List<int[]> diffIndices = new ArrayList<>();

        for(int i=0; i<stop; i++) {
            int[] indices = a.shape.getNdIndices(i);

            if(!a.get(indices).equals(b.get(indices))) {
                System.out.printf("Difference at %s: %s, %s\n",
                        Arrays.toString(a.shape.getNdIndices(i)),
                        a.get(indices),
                        b.get(indices));
                diffIndices.add(indices);
            }
        }

        return diffIndices;
    }
}

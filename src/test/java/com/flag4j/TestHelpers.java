package com.flag4j;

import com.flag4j.core.MatrixMixin;

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
}

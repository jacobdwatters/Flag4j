package com.flag4j;

import com.flag4j.core.MatrixMixin;

public class TestHelpers {

    public static void printAsNumpyArray(Object... args) {
        int name = 65;

        for(Object arg : args) {
            if(arg instanceof MatrixMixin) {
                printAsNumpyArray(Character.valueOf((char) name++).toString(), (MatrixMixin<?, ?, ?, ?, ?, ?, ?, ?>) arg);
            } else {
                System.out.print(arg.toString());
            }
        }
    }

    private static <T extends MatrixMixin<?, ?, ?, ?, ?, ?, ?, ?>> void printAsNumpyArray(String name, T A) {
        System.out.println(name + " = np.array([");

        for(int i=0; i<A.numRows(); i++) {
            System.out.print("\t[");
            for(int j=0; j<A.numCols(); j++) {
                System.out.print(A.get(i, j));

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
}

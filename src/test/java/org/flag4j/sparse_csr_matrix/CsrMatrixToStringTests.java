package org.flag4j.sparse_csr_matrix;

import org.flag4j.arrays_old.sparse.CooMatrixOld;
import org.flag4j.arrays_old.sparse.CsrMatrixOld;
import org.flag4j.core.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CsrMatrixToStringTests {
    static CsrMatrixOld A;
    static Shape aShape;
    static double[] nnz;
    static int[] colIndices;
    static int[] rowIndices;
    String exp = "";

    static void build() {
        A = new CooMatrixOld(aShape, nnz, rowIndices, colIndices).toCsr();
    }


    @Test
    void toStringTests() {
        // ------------------ Sub-case 1 ------------------
        aShape = new Shape(150, 2256);
        nnz = new double[]{1, 14.235, 239034, -882334.348, 15.235, 1.5342};
        rowIndices = new int[]{0, 0, 1, 121, 141, 149};
        colIndices = new int[]{150, 2500, 14, 15, 892, 156};
        exp = """
                Full Shape: (150, 2256)
                Non-zero entries: [ 1  14.235  239034  -882334.348  15.235  1.5342 ]
                Row Pointers: [0, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6]
                Col Indices: [150, 2500, 14, 15, 892, 156]""";
        build();

        assertEquals(exp, A.toString());

        // ------------------ Sub-case 2 ------------------
        aShape = new Shape(12, 12);
        nnz = new double[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
        rowIndices = new int[]{0, 0, 0, 0, 1, 1, 1,  2, 2, 2, 2, 2, 2,  2,  3, 5, 5, 5,  8, 8, 8, 8, 9, 9, 9};
        colIndices = new int[]{0, 1, 2, 9, 5, 6, 11, 0, 2, 7, 8, 9, 10, 11, 6, 2, 9, 11, 0, 1, 5, 7, 8, 9, 11};
        exp = """
                Full Shape: (12, 12)
                Non-zero entries: [ 1  2  3  4  5  6  7  8  9  ...  25 ]
                Row Pointers: [0, 4, 7, 14, 15, 15, 18, 18, 18, 22, 25, 25, 25]
                Col Indices: [0, 1, 2, 9, 5, 6, 11, 0, 2, 7, 8, 9, 10, 11, 6, 2, 9, 11, 0, 1, 5, 7, 8, 9, 11]""";
        build();

        assertEquals(exp, A.toString());
    }
}

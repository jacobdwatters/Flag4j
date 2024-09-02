package org.flag4j.sparse_csr_complex_matrix;

import org.flag4j.arrays_old.sparse.CooCMatrixOld;
import org.flag4j.arrays_old.sparse.CsrCMatrixOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.arrays.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class CsrCMatrixToStringTests {
    static CsrCMatrixOld A;
    static Shape aShape;
    static CNumber[] nnz;
    static int[] colIndices;
    static int[] rowIndices;
    String exp = "";

    static void build() {
        A = new CooCMatrixOld(aShape, nnz, rowIndices, colIndices).toCsr();
    }


    @Test
    void toStringTests() {
        // ------------------ Sub-case 1 ------------------
        aShape = new Shape(150, 2256);
        nnz = new CNumber[]{new CNumber(1.325, 9.2), new CNumber(-6, 1), new CNumber(23),
                new CNumber(0, 34615), new CNumber(0, -1), new CNumber(25, 1)};
        rowIndices = new int[]{0, 0, 1, 121, 141, 149};
        colIndices = new int[]{150, 2500, 14, 15, 892, 156};
        exp = """
                Full Shape: (150, 2256)
                Non-zero entries: [ 1.325 + 9.2i  -6 + i  23  34615i  -i  25 + i ]
                Row Pointers: [0, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6]
                Col Indices: [150, 2500, 14, 15, 892, 156]""";
        build();

        assertEquals(exp, A.toString());

        // ------------------ Sub-case 2 ------------------
        aShape = new Shape(12, 12);
        nnz = new CNumber[]{
                new CNumber(1), new CNumber(2), new CNumber(3), new CNumber(4), new CNumber(5),
                new CNumber(6), new CNumber(7), new CNumber(8), new CNumber(9), new CNumber(10),
                new CNumber(11), new CNumber(12), new CNumber(13), new CNumber(14), new CNumber(15), 
                new CNumber(16), new CNumber(17), new CNumber(18), new CNumber(19), new CNumber(20), 
                new CNumber(21), new CNumber(22), new CNumber(23), new CNumber(24), new CNumber(25)};
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

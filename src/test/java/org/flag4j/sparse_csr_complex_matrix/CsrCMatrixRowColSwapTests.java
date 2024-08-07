package org.flag4j.sparse_csr_complex_matrix;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CsrCMatrixRowColSwapTests {

    CsrCMatrix A;
    CsrCMatrix exp;
    CNumber[][] aEntries;
    CNumber[][] expEntries;

    @Test
    void rowSwapTests() {
        // ---------------------- Sub-case new CNumber(1) ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(10, -9.1), new CNumber(20, 1.5), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(30, 7.2), new CNumber(0), new CNumber(40), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(50, 56), new CNumber(60.1, -15), new CNumber(70.2, 15.34),
                        new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(80,2.1)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new CNumber[][]{
                {new CNumber(10, -9.1), new CNumber(20, 1.5), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(50, 56), new CNumber(60.1, -15), new CNumber(70.2, 15.34),
                        new CNumber(0)},
                {new CNumber(0), new CNumber(30, 7.2), new CNumber(0), new CNumber(40), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(80, 2.1)}};
        exp = new CMatrix(expEntries).toCsr();

        assertEquals(exp, A.swapRows(1, 2));

        // ---------------------- Sub-case new CNumber(2) ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(10, -9.1), new CNumber(20, 1.5), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(30, 7.2), new CNumber(0), new CNumber(40), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(50, 56), new CNumber(60, -15), new CNumber(70, 15.34), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(80, 2.1)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new CNumber[][]{
                {new CNumber(10, -9.1), new CNumber(20, 1.5), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(50, 56), new CNumber(60, -15), new CNumber(70, 15.34), new CNumber(0)},
                {new CNumber(0), new CNumber(30, 7.2), new CNumber(0), new CNumber(40), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(80, 2.1)}};
        exp = new CMatrix(expEntries).toCsr();

        assertEquals(exp, A.swapRows(2, 1));

        // ---------------------- Sub-case new CNumber(3) ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(10, -9.1), new CNumber(20, 1.5), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(30, 7.2), new CNumber(0), new CNumber(40), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(50, 56), new CNumber(60, -15), new CNumber(70, 15.34), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(80, 2.1)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(0), new CNumber(50, 56), new CNumber(60, -15), new CNumber(70, 15.34), new CNumber(0)},
                {new CNumber(0), new CNumber(30, 7.2), new CNumber(0), new CNumber(40), new CNumber(0), new CNumber(0)},
                {new CNumber(10, -9.1), new CNumber(20, 1.5), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(80, 2.1)}};
        exp = new CMatrix(expEntries).toCsr();

        assertEquals(exp, A.swapRows(2, 0));

        // ---------------------- Sub-case new CNumber(4) ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(10, -9.1), new CNumber(20, 1.5), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(30, 7.2), new CNumber(0), new CNumber(40), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(50, 56), new CNumber(60, -15), new CNumber(70, 15.34), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(80, 2.1)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(80, 2.1)},
                {new CNumber(0), new CNumber(30, 7.2), new CNumber(0), new CNumber(40), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(50, 56), new CNumber(60, -15), new CNumber(70, 15.34), new CNumber(0)},
                {new CNumber(10, -9.1), new CNumber(20, 1.5), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)}};
        exp = new CMatrix(expEntries).toCsr();

        assertEquals(exp, A.swapRows(0, 3));

        // ---------------------- Sub-case new CNumber(5) ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(10, -9.1), new CNumber(20, 1.5), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(30, 7.2), new CNumber(0), new CNumber(40), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(50, 56), new CNumber(60, -15), new CNumber(70, 15.34), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(80, 2.1)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new CNumber[][]{
                {new CNumber(10, -9.1), new CNumber(20, 1.5), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(30, 7.2), new CNumber(0), new CNumber(40), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(80, 2.1)},
                {new CNumber(0), new CNumber(0), new CNumber(50, 56), new CNumber(60, -15), new CNumber(70, 15.34), new CNumber(0)}};
        exp = new CMatrix(expEntries).toCsr();

        assertEquals(exp, A.swapRows(3, 2));

        // ---------------------- Sub-case new CNumber(6) ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(10, -9.1), new CNumber(20, 1.5), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(30, 7.2), new CNumber(0), new CNumber(40), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(50, 56), new CNumber(60, -15), new CNumber(70, 15.34), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(80, 2.1)}};
        A = new CMatrix(aEntries).toCsr();

        assertThrows(IndexOutOfBoundsException.class, ()->A.swapRows(-1, 2));
        assertThrows(IndexOutOfBoundsException.class, ()->A.swapRows(1, 145));
        assertThrows(IndexOutOfBoundsException.class, ()->A.swapRows(0, 145));
    }


    @Test
    void colSwapTests() {
        // ---------------------- Sub-case new CNumber(1) ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(10, -9.1), new CNumber(20, 1.5), new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0)},
                {new CNumber(0),  new CNumber(30, 7.2), new CNumber(0),  new CNumber(40), new CNumber(0),  new CNumber(0)},
                {new CNumber(0),  new CNumber(0),  new CNumber(50, 56), new CNumber(60, -15), new CNumber(70, 15.34), new CNumber(0)},
                {new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(80, 2.1)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new CNumber[][]{
                {new CNumber(0),  new CNumber(20, 1.5), new CNumber(0),  new CNumber(10, -9.1), new CNumber(0),  new CNumber(0)},
                {new CNumber(40), new CNumber(30, 7.2), new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0)},
                {new CNumber(60, -15), new CNumber(0),  new CNumber(50, 56), new CNumber(0),  new CNumber(70, 15.34), new CNumber(0)},
                {new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(80, 2.1)}};
        exp = new CMatrix(expEntries).toCsr();

        assertEquals(exp, A.swapCols(0, 3));

        // ---------------------- Sub-case new CNumber(2) ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(10, -9.1), new CNumber(20, 1.5), new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0)},
                {new CNumber(0),  new CNumber(30, 7.2), new CNumber(0),  new CNumber(40), new CNumber(0),  new CNumber(0)},
                {new CNumber(0),  new CNumber(0),  new CNumber(50, 56), new CNumber(60, -15), new CNumber(70, 15.34), new CNumber(0)},
                {new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(80, 2.1)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new CNumber[][]{
                {new CNumber(0),  new CNumber(20, 1.5), new CNumber(0),  new CNumber(10, -9.1), new CNumber(0),  new CNumber(0)},
                {new CNumber(40), new CNumber(30, 7.2), new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0)},
                {new CNumber(60, -15), new CNumber(0),  new CNumber(50, 56), new CNumber(0),  new CNumber(70, 15.34), new CNumber(0)},
                {new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(80, 2.1)}};
        exp = new CMatrix(expEntries).toCsr();

        assertEquals(exp, A.swapCols(3, 0));

        // ---------------------- Sub-case new CNumber(3) ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(10, -9.1), new CNumber(20, 1.5), new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0)},
                {new CNumber(0),  new CNumber(30, 7.2), new CNumber(0),  new CNumber(40), new CNumber(0),  new CNumber(0)},
                {new CNumber(0),  new CNumber(0),  new CNumber(50, 56), new CNumber(60, -15), new CNumber(70, 15.34), new CNumber(0)},
                {new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(80, 2.1)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new CNumber[][]{
                {new CNumber(10, -9.1), new CNumber(0),  new CNumber(0),  new CNumber(20, 1.5), new CNumber(0),  new CNumber(0)},
                {new CNumber(0),  new CNumber(40), new CNumber(0),  new CNumber(30, 7.2), new CNumber(0),  new CNumber(0)},
                {new CNumber(0),  new CNumber(60, -15), new CNumber(50, 56), new CNumber(0),  new CNumber(70, 15.34), new CNumber(0)},
                {new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(80, 2.1)}};
        exp = new CMatrix(expEntries).toCsr();

        assertEquals(exp, A.swapCols(1, 3));

        // ---------------------- Sub-case new CNumber(4) ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(10, -9.1), new CNumber(20, 1.5), new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0) },
                {new CNumber(0),  new CNumber(30, 7.2), new CNumber(0),  new CNumber(40), new CNumber(0),  new CNumber(0) },
                {new CNumber(0),  new CNumber(0),  new CNumber(50, 56), new CNumber(60, -15), new CNumber(70, 15.34), new CNumber(0) },
                {new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(80, 2.1)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new CNumber[][]{
                {new CNumber(0),  new CNumber(20, 1.5), new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(10, -9.1)},
                {new CNumber(0),  new CNumber(30, 7.2), new CNumber(0),  new CNumber(40), new CNumber(0),  new CNumber(0) },
                {new CNumber(0),  new CNumber(0),  new CNumber(50, 56), new CNumber(60, -15), new CNumber(70, 15.34), new CNumber(0) },
                {new CNumber(80, 2.1), new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0) }};
        exp = new CMatrix(expEntries).toCsr();

        assertEquals(exp, A.swapCols(0, 5));

        // ---------------------- Sub-case new CNumber(5) ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(10, -9.1), new CNumber(20, 1.5), new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0) },
                {new CNumber(0),  new CNumber(30, 7.2), new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0) },
                {new CNumber(0),  new CNumber(40), new CNumber(50, 56), new CNumber(60, -15), new CNumber(70, 15.34), new CNumber(0) },
                {new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(80, 2.1),  new CNumber(0), new CNumber(90)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new CNumber[][]{
                {new CNumber(10, -9.1), new CNumber(0),  new CNumber(0),  new CNumber(20, 1.5), new CNumber(0),  new CNumber(0) },
                {new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(30, 7.2), new CNumber(0),  new CNumber(0) },
                {new CNumber(0),  new CNumber(60, -15), new CNumber(50, 56), new CNumber(40), new CNumber(70, 15.34), new CNumber(0) },
                {new CNumber(0),  new CNumber(80, 2.1), new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(90)}};
        exp = new CMatrix(expEntries).toCsr();

        assertEquals(exp, A.swapCols(1, 3));

        // ---------------------- Sub-case new CNumber(6) ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(10, -9.1), new CNumber(20, 1.5), new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0) },
                {new CNumber(0),  new CNumber(30, 7.2), new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0) },
                {new CNumber(0),  new CNumber(40), new CNumber(50, 56), new CNumber(60, -15), new CNumber(70, 15.34), new CNumber(0) },
                {new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(80, 2.1),  new CNumber(0), new CNumber(90)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new CNumber[][]{
                {new CNumber(10, -9.1), new CNumber(20, 1.5), new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0) },
                {new CNumber(0),  new CNumber(30, 7.2), new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0) },
                {new CNumber(0),  new CNumber(40), new CNumber(50, 56), new CNumber(0),  new CNumber(70, 15.34), new CNumber(60, -15) },
                {new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(90), new CNumber(0), new CNumber(80, 2.1)}};
        exp = new CMatrix(expEntries).toCsr();

        assertEquals(exp, A.swapCols(3, 5));

        // ---------------------- Sub-case new CNumber(7) ----------------------
        aEntries = new CNumber[][]{
                {new CNumber(10, -9.1), new CNumber(20, 1.5), new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0) },
                {new CNumber(0),  new CNumber(30, 7.2), new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0) },
                {new CNumber(0),  new CNumber(40), new CNumber(50, 56), new CNumber(60, -15), new CNumber(70, 15.34), new CNumber(0) },
                {new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(80, 2.1),  new CNumber(0), new CNumber(90)}};
        A = new CMatrix(aEntries).toCsr();
        expEntries = new CNumber[][]{
                {new CNumber(20, 1.5), new CNumber(10, -9.1), new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0) },
                {new CNumber(30, 7.2), new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(0) },
                {new CNumber(40), new CNumber(0),  new CNumber(50, 56), new CNumber(60, -15), new CNumber(70, 15.34), new CNumber(0) },
                {new CNumber(0),  new CNumber(0),  new CNumber(0),  new CNumber(80, 2.1),  new CNumber(0), new CNumber(90)}};
        exp = new CMatrix(expEntries).toCsr();

        assertEquals(exp, A.swapCols(0, 1));
    }
}

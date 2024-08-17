package org.flag4j.sparse_csr_complex_matrix;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.sparse.CsrCMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CsrCMatrixGetSetTests {

    static CsrCMatrix A;
    static CNumber[][] aEntries;
    static CsrCMatrix exp;
    static CNumber[][] expEntries;


    @Test
    void getTests() {
        // -------------------------- Sub-case 1 --------------------------
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(1, -72.1), new CNumber(0), new CNumber(0), new CNumber(4), new CNumber(0),
                        new CNumber(0)},
                {new CNumber(0), new CNumber(9.1), new CNumber(0), new CNumber(0), new CNumber(-1.4, 34.1), new CNumber(0),
                        new CNumber(0)},
                {new CNumber(0), new CNumber(0, 1.5), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(4, -3), new CNumber(0), new CNumber(801.4, 15),
                        new CNumber(-15, 1)}};
        A = new CMatrixOld(aEntries).toCsr();

        assertEquals(new CNumber(0), A.get(0, 0));
        assertEquals(new CNumber(1, -72.1), A.get(0, 1));
        assertEquals(new CNumber(0), A.get(0, 2));
        assertEquals(new CNumber(0), A.get(0, 3));
        assertEquals(new CNumber(4), A.get(0, 4));
        assertEquals(new CNumber(0), A.get(0, 5));
        assertEquals(new CNumber(0), A.get(0, 6));

        assertEquals(new CNumber(0), A.get(1, 0));
        assertEquals(new CNumber(9.1), A.get(1, 1));
        assertEquals(new CNumber(0), A.get(1, 2));
        assertEquals(new CNumber(0), A.get(1, 3));
        assertEquals(new CNumber(-1.4, 34.1), A.get(1, 4));
        assertEquals(new CNumber(0), A.get(1, 5));
        assertEquals(new CNumber(0), A.get(1, 6));

        assertEquals(new CNumber(0), A.get(2, 0));
        assertEquals(new CNumber(0, 1.5), A.get(2, 1));
        assertEquals(new CNumber(0), A.get(2, 2));
        assertEquals(new CNumber(0), A.get(2, 3));
        assertEquals(new CNumber(0), A.get(2, 4));
        assertEquals(new CNumber(0), A.get(2, 5));
        assertEquals(new CNumber(0), A.get(2, 6));

        assertEquals(new CNumber(0), A.get(3, 0));
        assertEquals(new CNumber(0), A.get(3, 1));
        assertEquals(new CNumber(0), A.get(3, 2));
        assertEquals(new CNumber(4, -3), A.get(3, 3));
        assertEquals(new CNumber(0), A.get(3, 4));
        assertEquals(new CNumber(801.4, 15), A.get(3, 5));
        assertEquals(new CNumber(-15, 1), A.get(3, 6));

        // -------------------------- Sub-case 2 --------------------------
        assertThrows(IndexOutOfBoundsException.class, ()->A.get(-1, 0));
        assertThrows(IndexOutOfBoundsException.class, ()->A.get(0, -1));
        assertThrows(IndexOutOfBoundsException.class, ()->A.get(4, 0));
        assertThrows(IndexOutOfBoundsException.class, ()->A.get(0, 7));
        assertThrows(IndexOutOfBoundsException.class, ()->A.get(20, 15));
    }


    @Test
    void setTests() {
        aEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(1, -72.1), new CNumber(0), new CNumber(0), new CNumber(4), new CNumber(0),
                        new CNumber(0)},
                {new CNumber(0), new CNumber(9.1), new CNumber(0), new CNumber(0), new CNumber(-1.4, 34.1), new CNumber(0),
                        new CNumber(0)},
                {new CNumber(0), new CNumber(0, 1.5), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(4, -3), new CNumber(0), new CNumber(801.4, 15),
                        new CNumber(-15, 1)}};
        A = new CMatrixOld(aEntries).toCsr();

        // -------------------------- Sub-case 1 --------------------------
        expEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(1, -72.1), new CNumber(0), new CNumber(0),
                        new CNumber(4), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(9.1), new CNumber(0), new CNumber(0),
                        new CNumber(9,-1.25), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0, 1.5), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(4, -3),
                        new CNumber(0), new CNumber(801.4, 15), new CNumber(-15, 1)}};
        exp = new CMatrixOld(expEntries).toCsr();
        assertEquals(exp, A.set(new CNumber(9,-1.25), 1, 4));

        // -------------------------- Sub-case 2 --------------------------
        expEntries = new CNumber[][]{
                {new CNumber(6, -2), new CNumber(1, -72.1), new CNumber(0), new CNumber(0),
                        new CNumber(4), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(9.1), new CNumber(0), new CNumber(0),
                        new CNumber(-1.4, 34.1), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0, 1.5), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(4, -3),
                        new CNumber(0), new CNumber(801.4, 15), new CNumber(-15, 1)}};
        exp = new CMatrixOld(expEntries).toCsr();
        assertEquals(exp, A.set(new CNumber(6, -2), 0, 0));

        // -------------------------- Sub-case 3 --------------------------
        expEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(1, -72.1), new CNumber(0), new CNumber(0),
                        new CNumber(4), new CNumber(0), new CNumber(5)},
                {new CNumber(0), new CNumber(9.1), new CNumber(0), new CNumber(0),
                        new CNumber(-1.4, 34.1), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0, 1.5), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(4, -3),
                        new CNumber(0), new CNumber(801.4, 15), new CNumber(-15, 1)}};
        exp = new CMatrixOld(expEntries).toCsr();
        assertEquals(exp, A.set(5, 0, 6));

        // -------------------------- Sub-case 4 --------------------------
        expEntries = new CNumber[][]{
                {new CNumber(0), new CNumber(1, -72.1), new CNumber(0), new CNumber(0),
                        new CNumber(4), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(9.1), new CNumber(0), new CNumber(0),
                        new CNumber(-1.4, 34.1), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0, 1.5), new CNumber(0), new CNumber(0),
                        new CNumber(0), new CNumber(0), new CNumber(0)},
                {new CNumber(0), new CNumber(0), new CNumber(0), new CNumber(4, -3),
                        new CNumber(-5900.1, 2352.26221), new CNumber(801.4, 15), new CNumber(-15, 1)}};
        exp = new CMatrixOld(expEntries).toCsr();
        assertEquals(exp, A.set(new CNumber(-5900.1, 2352.26221), 3, 4));

        // -------------------------- Sub-case 5 --------------------------
        assertThrows(IndexOutOfBoundsException.class, ()->A.set(1.3, -1, 0));
        assertThrows(IndexOutOfBoundsException.class, ()->A.set(1.3, 0, -1));
        assertThrows(IndexOutOfBoundsException.class, ()->A.set(1.3, 4, 0));
        assertThrows(IndexOutOfBoundsException.class, ()->A.set(1.3, 0, 7));
        assertThrows(IndexOutOfBoundsException.class, ()->A.set(1.3, 20, 15));
    }
}

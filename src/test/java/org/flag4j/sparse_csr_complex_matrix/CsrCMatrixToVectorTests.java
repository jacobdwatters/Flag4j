package org.flag4j.sparse_csr_complex_matrix;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.sparse.CooCVector;
import org.flag4j.arrays_old.sparse.CsrCMatrix;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CsrCMatrixToVectorTests {
    CsrCMatrix A;
    CNumber[][] aEntries;

    CooCVector exp;
    CNumber[] expEntries;

    @Test
    void toVectorTests() {
        // ------------------------- Sub-case 1 -------------------------
        aEntries = new CNumber[][]{{new CNumber(1.23, -9.25), CNumber.ZERO, CNumber.ZERO,
                CNumber.ZERO, CNumber.ZERO, new CNumber(1.526, -3.1),
                CNumber.ZERO, CNumber.ZERO, new CNumber(0, 1.2)}};
        A = new CMatrixOld(aEntries).toCsr();
        expEntries = new CNumber[]{new CNumber(1.23, -9.25), CNumber.ZERO, CNumber.ZERO,
                CNumber.ZERO, CNumber.ZERO, new CNumber(1.526, -3.1),
                CNumber.ZERO, CNumber.ZERO, new CNumber(0, 1.2)};
        exp = new CVectorOld(expEntries).toCoo();

        assertEquals(exp, A.toVector());

        // ------------------------- Sub-case 2 -------------------------
        aEntries = new CNumber[][]{{new CNumber(1.23, -9.25)}, {CNumber.ZERO}, {CNumber.ZERO},
                {CNumber.ZERO}, {CNumber.ZERO}, {new CNumber(1.526, -3.1)},
                {CNumber.ZERO}, {CNumber.ZERO}, {new CNumber(0, 1.2)}};
        A = new CMatrixOld(aEntries).toCsr();
        expEntries = new CNumber[]{new CNumber(1.23, -9.25), CNumber.ZERO, CNumber.ZERO,
                CNumber.ZERO, CNumber.ZERO, new CNumber(1.526, -3.1),
                CNumber.ZERO, CNumber.ZERO, new CNumber(0, 1.2)};
        exp = new CVectorOld(expEntries).toCoo();

        assertEquals(exp, A.toVector());

        // ------------------------- Sub-case 3 -------------------------
        aEntries = new CNumber[][]{
                {new CNumber(1.23, -9.25), CNumber.ZERO},
                {CNumber.ZERO, CNumber.ZERO},
                {CNumber.ZERO, new CNumber(1.526, -3.1)},
                {CNumber.ZERO, CNumber.ZERO},
                {new CNumber(0, 1.2), CNumber.ZERO}};
        A = new CMatrixOld(aEntries).toCsr();
        expEntries = new CNumber[]{new CNumber(1.23, -9.25), CNumber.ZERO, CNumber.ZERO,
                CNumber.ZERO, CNumber.ZERO, new CNumber(1.526, -3.1),
                CNumber.ZERO, CNumber.ZERO, new CNumber(0, 1.2), CNumber.ZERO};
        exp = new CVectorOld(expEntries).toCoo();

        assertEquals(exp, A.toVector());
    }
}

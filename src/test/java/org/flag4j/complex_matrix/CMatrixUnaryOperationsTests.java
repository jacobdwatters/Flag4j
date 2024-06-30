package org.flag4j.complex_matrix;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.dense.CMatrix;
import org.flag4j.dense.Matrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CMatrixUnaryOperationsTests {

    CNumber[][] aEntries, expEntries;
    double[][] expReEntries;

    CMatrix A, exp;
    Matrix expRe;
    CNumber expComplex;


    @Test
    void sumTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1), new CNumber(32.4), new CNumber(394728.1)},
                {new CNumber(-9841, -85.13), new CNumber(0, 84.1), new CNumber(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expComplex = new CNumber(234.66, -9923.1).add(new CNumber(32.4)).add(new CNumber(394728.1))
                .add(new CNumber(-9841, -85.13)).add(new CNumber(0, 84.1)).add(new CNumber(-5.234, 234));

        assertEquals(expComplex, A.sum());
    }


    @Test
    void sqrtTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1), new CNumber(32.4), new CNumber(394728.1)},
                {new CNumber(-9841, -85.13), new CNumber(0, 84.1), new CNumber(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {CNumber.sqrt(new CNumber(234.66, -9923.1)), CNumber.sqrt(new CNumber(32.4)), CNumber.sqrt(new CNumber(394728.1))},
                {CNumber.sqrt(new CNumber(-9841, -85.13)), CNumber.sqrt(new CNumber(0, 84.1)), CNumber.sqrt(new CNumber(-5.234, 234))}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.sqrt());
    }


    @Test
    void absTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1), new CNumber(32.4), new CNumber(394728.1)},
                {new CNumber(-9841, -85.13), new CNumber(0, 84.1), new CNumber(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expReEntries = new double[][]{
                {new CNumber(234.66, -9923.1).mag(), new CNumber(32.4).mag(), new CNumber(394728.1).mag()},
                {new CNumber(-9841, -85.13).mag(), new CNumber(0, 84.1).mag(), new CNumber(-5.234, 234).mag()}};
        expRe = new Matrix(expReEntries);

        assertEquals(expRe, A.abs());
    }


    @Test
    void transposeTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1), new CNumber(32.4), new CNumber(394728.1)},
                {new CNumber(-9841, -85.13), new CNumber(0, 84.1), new CNumber(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1), new CNumber(-9841, -85.13)},
                {new CNumber(32.4), new CNumber(0, 84.1)},
                {new CNumber(394728.1), new CNumber(-5.234, 234)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.transpose());


        // -------------------- Sub-case 2 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1), new CNumber(32.4), new CNumber(394728.1)},
                {new CNumber(-9841, -85.13), new CNumber(0, 84.1), new CNumber(-5.234, 234)},
                {new CNumber(234.56, 9.4), new CNumber(0.43467, 5.2), new CNumber(234.7, -0.412)}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1), new CNumber(-9841, -85.13), new CNumber(234.56, 9.4)},
                {new CNumber(32.4), new CNumber(0, 84.1), new CNumber(0.43467, 5.2)},
                {new CNumber(394728.1), new CNumber(-5.234, 234), new CNumber(234.7, -0.412)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.transpose());

        // -------------------- Sub-case 3 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1), new CNumber(32.4)},
                {new CNumber(-9841, -85.13), new CNumber(0, 84.1)},
                {new CNumber(234.56, 9.4), new CNumber(0.43467, 5.2)}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1), new CNumber(-9841, -85.13), new CNumber(234.56, 9.4)},
                {new CNumber(32.4), new CNumber(0, 84.1), new CNumber(0.43467, 5.2)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.transpose());
    }


    @Test
    void hermationTransposeTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1), new CNumber(32.4), new CNumber(394728.1)},
                {new CNumber(-9841, -85.13), new CNumber(0, 84.1), new CNumber(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(234.66, 9923.1), new CNumber(-9841, 85.13)},
                {new CNumber(32.4), new CNumber(0, -84.1)},
                {new CNumber(394728.1), new CNumber(-5.234, -234)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.hermTranspose());

        // -------------------- Sub-case 2 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1), new CNumber(32.4), new CNumber(394728.1)},
                {new CNumber(-9841, -85.13), new CNumber(0, 84.1), new CNumber(-5.234, 234)},
                {new CNumber(234.56, 9.4), new CNumber(0.43467, 5.2), new CNumber(234.7, -0.412)}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(234.66, 9923.1), new CNumber(-9841, 85.13), new CNumber(234.56, -9.4)},
                {new CNumber(32.4), new CNumber(0, -84.1), new CNumber(0.43467, -5.2)},
                {new CNumber(394728.1), new CNumber(-5.234, -234), new CNumber(234.7, 0.412)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.hermTranspose());

        // -------------------- Sub-case 3 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1), new CNumber(32.4)},
                {new CNumber(-9841, -85.13), new CNumber(0, 84.1)},
                {new CNumber(234.56, 9.4), new CNumber(0.43467, 5.2)}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(234.66, 9923.1), new CNumber(-9841, 85.13), new CNumber(234.56, -9.4)},
                {new CNumber(32.4), new CNumber(0, -84.1), new CNumber(0.43467, -5.2)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.hermTranspose());
    }


    @Test
    void recipTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1), new CNumber(32.4), new CNumber(394728.1)},
                {new CNumber(-9841, -85.13), new CNumber(0, 84.1), new CNumber(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1).multInv(), new CNumber(32.4).multInv(), new CNumber(394728.1).multInv()},
                {new CNumber(-9841, -85.13).multInv(), new CNumber(0, 84.1).multInv(), new CNumber(-5.234, 234).multInv()}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.recip());
    }


    @Test
    void conjTestCase() {
        // -------------------- Sub-case 1 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1), new CNumber(32.4), new CNumber(394728.1)},
                {new CNumber(-9841, -85.13), new CNumber(0, 84.1), new CNumber(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(234.66, 9923.1), new CNumber(32.4), new CNumber(394728.1)},
                {new CNumber(-9841, 85.13), new CNumber(0, -84.1), new CNumber(-5.234, -234)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.conj());

        // -------------------- Sub-case 2 --------------------
        aEntries = new CNumber[][]{
                {new CNumber(234.66, 9923.1), new CNumber(-9841, 85.13)},
                {new CNumber(32.4), new CNumber(0, -84.1)},
                {new CNumber(394728.1), new CNumber(-5.234, -234)}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1), new CNumber(-9841, -85.13)},
                {new CNumber(32.4), new CNumber(0, 84.1)},
                {new CNumber(394728.1), new CNumber(-5.234, 234)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.conj());
    }
}

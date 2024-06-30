package org.flag4j.complex_matrix;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.dense.CMatrix;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class CMatrixScaleOperationsTests {

    CNumber[][] aEntries, expEntries;
    CMatrix A, exp;

    @Test
    void realScaleMultTestCase() {
        double scal;

        // ---------------------- Sub-case 1 ----------------------
        scal = 79.3419;
        aEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1), new CNumber(32.4), new CNumber(394728.1)},
                {new CNumber(-9841, -85.13), new CNumber(0, 84.1), new CNumber(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1).mult(scal), new CNumber(32.4).mult(scal), new CNumber(394728.1).mult(scal)},
                {new CNumber(-9841, -85.13).mult(scal), new CNumber(0, 84.1).mult(scal), new CNumber(-5.234, 234).mult(scal)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.mult(scal));

        // ---------------------- Sub-case 2 ----------------------
        scal = -2179.3419;
        aEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1), new CNumber(32.4), new CNumber(394728.1)},
                {new CNumber(-9841, -85.13), new CNumber(0, 84.1), new CNumber(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1).mult(scal), new CNumber(32.4).mult(scal), new CNumber(394728.1).mult(scal)},
                {new CNumber(-9841, -85.13).mult(scal), new CNumber(0, 84.1).mult(scal), new CNumber(-5.234, 234).mult(scal)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.mult(scal));
    }


    @Test
    void complexScaleMultTestCase() {
        CNumber scal;

        // ---------------------- Sub-case 1 ----------------------
        scal = new CNumber(9234.1, -923.1);
        aEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1), new CNumber(32.4), new CNumber(394728.1)},
                {new CNumber(-9841, -85.13), new CNumber(0, 84.1), new CNumber(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1).mult(scal), new CNumber(32.4).mult(scal), new CNumber(394728.1).mult(scal)},
                {new CNumber(-9841, -85.13).mult(scal), new CNumber(0, 84.1).mult(scal), new CNumber(-5.234, 234).mult(scal)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.mult(scal));

        // ---------------------- Sub-case 2 ----------------------
        scal = new CNumber(-0.000234, -923.1);
        aEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1), new CNumber(32.4), new CNumber(394728.1)},
                {new CNumber(-9841, -85.13), new CNumber(0, 84.1), new CNumber(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1).mult(scal), new CNumber(32.4).mult(scal), new CNumber(394728.1).mult(scal)},
                {new CNumber(-9841, -85.13).mult(scal), new CNumber(0, 84.1).mult(scal), new CNumber(-5.234, 234).mult(scal)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.mult(scal));
    }


    @Test
    void realScaleDivTestCase() {
        double scal;

        // ---------------------- Sub-case 1 ----------------------
        scal = 79.3419;
        aEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1), new CNumber(32.4), new CNumber(394728.1)},
                {new CNumber(-9841, -85.13), new CNumber(0, 84.1), new CNumber(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1).div(scal), new CNumber(32.4).div(scal), new CNumber(394728.1).div(scal)},
                {new CNumber(-9841, -85.13).div(scal), new CNumber(0, 84.1).div(scal), new CNumber(-5.234, 234).div(scal)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.div(scal));

        // ---------------------- Sub-case 2 ----------------------
        scal = -2179.3419;
        aEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1), new CNumber(32.4), new CNumber(394728.1)},
                {new CNumber(-9841, -85.13), new CNumber(0, 84.1), new CNumber(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1).div(scal), new CNumber(32.4).div(scal), new CNumber(394728.1).div(scal)},
                {new CNumber(-9841, -85.13).div(scal), new CNumber(0, 84.1).div(scal), new CNumber(-5.234, 234).div(scal)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.div(scal));
    }


    @Test
    void complexScaleDivTestCase() {
        CNumber scal;

        // ---------------------- Sub-case 1 ----------------------
        scal = new CNumber(9234.1, -923.1);
        aEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1), new CNumber(32.4), new CNumber(394728.1)},
                {new CNumber(-9841, -85.13), new CNumber(0, 84.1), new CNumber(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1).div(scal), new CNumber(32.4).div(scal), new CNumber(394728.1).div(scal)},
                {new CNumber(-9841, -85.13).div(scal), new CNumber(0, 84.1).div(scal), new CNumber(-5.234, 234).div(scal)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.div(scal));

        // ---------------------- Sub-case 2 ----------------------
        scal = new CNumber(-0.000234, -923.1);
        aEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1), new CNumber(32.4), new CNumber(394728.1)},
                {new CNumber(-9841, -85.13), new CNumber(0, 84.1), new CNumber(-5.234, 234)}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(234.66, -9923.1).div(scal), new CNumber(32.4).div(scal), new CNumber(394728.1).div(scal)},
                {new CNumber(-9841, -85.13).div(scal), new CNumber(0, 84.1).div(scal), new CNumber(-5.234, 234).div(scal)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.div(scal));
    }
}


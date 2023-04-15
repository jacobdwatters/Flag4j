package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CMatrixRoundTests {

    int tol;
    CNumber[][] aEntries, expEntries;
    CMatrix A, exp;

    @Test
    void simpleRoundTest() {
        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(23.5, Double.NEGATIVE_INFINITY), new CNumber(56, -93.1), new CNumber(3.455, 1.54)},
                {new CNumber(5, Double.NaN), new CNumber(-9854.333, 0.000003), new CNumber(Double.POSITIVE_INFINITY, 9.99)}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(24, Double.NEGATIVE_INFINITY), new CNumber(56, -93), new CNumber(3, 2)},
                {new CNumber(5, Double.NaN), new CNumber(-9854, 00), new CNumber(Double.POSITIVE_INFINITY, 10)}};
        exp = new CMatrix(expEntries);

        CMatrix B = A.round();

        assertEquals(exp.shape, B.shape);

        for(int i=0; i<exp.entries.length; i++) {
            if(!exp.entries[i].isNaN()) {
                assertEquals(exp.entries[i], B.entries[i]);
            } else {
                double expRe = exp.entries[i].re;
                double expIm = exp.entries[i].im;

                if(Double.isNaN(expRe)) {
                    assertTrue(Double.isNaN(B.entries[i].re));
                } else {
                    assertEquals(expRe, B.entries[i].re);
                }

                if(Double.isNaN(expIm)) {
                    assertTrue(Double.isNaN(B.entries[i].im));
                } else {
                    assertEquals(expIm, B.entries[i].im);
                }
            }
        }
    }


    @Test
    void roundTest() {
        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(23.5884, Double.NEGATIVE_INFINITY), new CNumber(56, -93.134), new CNumber(3.4557734, 1.54)},
                {new CNumber(5.0043, Double.NaN), new CNumber(-9854.333, 0.000003), new CNumber(Double.POSITIVE_INFINITY, 9.7999)}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(23.588, Double.NEGATIVE_INFINITY), new CNumber(56, -93.134), new CNumber(3.456, 1.54)},
                {new CNumber(5.004, Double.NaN), new CNumber(-9854.333, 0), new CNumber(Double.POSITIVE_INFINITY, 9.8)}};
        exp = new CMatrix(expEntries);
        int tol = 3;
        CMatrix B = A.round(tol);

        assertEquals(exp.shape, B.shape);

        for(int i=0; i<exp.entries.length; i++) {
            if(!exp.entries[i].isNaN()) {
                assertEquals(exp.entries[i], B.entries[i]);
            } else {
                double expRe = exp.entries[i].re;
                double expIm = exp.entries[i].im;

                if(Double.isNaN(expRe)) {
                    assertTrue(Double.isNaN(B.entries[i].re));
                } else {
                    assertEquals(expRe, B.entries[i].re);
                }

                if(Double.isNaN(expIm)) {
                    assertTrue(Double.isNaN(B.entries[i].im));
                } else {
                    assertEquals(expIm, B.entries[i].im);
                }
            }
        }
    }


    @Test
    void roundToZeroTest() {
        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(23.5, Double.NEGATIVE_INFINITY), new CNumber(0, -0.5e-12), new CNumber(3.455, 1.54)},
                {new CNumber(5, Double.NaN), new CNumber(-34.5e-14, 34.5e-14), new CNumber(Double.POSITIVE_INFINITY, 9.99)}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(23.5, Double.NEGATIVE_INFINITY), new CNumber(0), new CNumber(3.455, 1.54)},
                {new CNumber(5, Double.NaN), new CNumber(0), new CNumber(Double.POSITIVE_INFINITY, 9.99)}};
        exp = new CMatrix(expEntries);

        CMatrix B = A.roundToZero();

        assertEquals(exp.shape, B.shape);

        for(int i=0; i<exp.entries.length; i++) {
            if(!exp.entries[i].isNaN()) {
                assertEquals(exp.entries[i], B.entries[i]);
            } else {
                double expRe = exp.entries[i].re;
                double expIm = exp.entries[i].im;

                if(Double.isNaN(expRe)) {
                    assertTrue(Double.isNaN(B.entries[i].re));
                } else {
                    assertEquals(expRe, B.entries[i].re);
                }

                if(Double.isNaN(expIm)) {
                    assertTrue(Double.isNaN(B.entries[i].im));
                } else {
                    assertEquals(expIm, B.entries[i].im);
                }
            }
        }
    }


    @Test
    void roundToZeroTolTest() {
        // ------------------------ Sub-case 1 ------------------------
        aEntries = new CNumber[][]{
                {new CNumber(23.5, Double.NEGATIVE_INFINITY), new CNumber(0, -0.5e-12), new CNumber(1, 1.54)},
                {new CNumber(5, Double.NaN), new CNumber(-34.5e-14, 34.5e-14), new CNumber(Double.POSITIVE_INFINITY, 9.99)}};
        A = new CMatrix(aEntries);
        expEntries = new CNumber[][]{
                {new CNumber(23.5, Double.NEGATIVE_INFINITY), new CNumber(0), new CNumber(0)},
                {new CNumber(5, Double.NaN), new CNumber(0), new CNumber(Double.POSITIVE_INFINITY, 9.99)}};
        exp = new CMatrix(expEntries);
        double tol = 4.3;

        CMatrix B = A.roundToZero(tol);

        assertEquals(exp.shape, B.shape);

        for(int i=0; i<exp.entries.length; i++) {
            if(!exp.entries[i].isNaN()) {
                assertEquals(exp.entries[i], B.entries[i]);
            } else {
                double expRe = exp.entries[i].re;
                double expIm = exp.entries[i].im;

                if(Double.isNaN(expRe)) {
                    assertTrue(Double.isNaN(B.entries[i].re));
                } else {
                    assertEquals(expRe, B.entries[i].re);
                }

                if(Double.isNaN(expIm)) {
                    assertTrue(Double.isNaN(B.entries[i].im));
                } else {
                    assertEquals(expIm, B.entries[i].im);
                }
            }
        }
    }
}

package com.flag4j;

import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class ComplexConversionTests {

    Matrix A;
    double[][] aEntries;
    CMatrix B, exp, act;
    CNumber[][] expEntries;


    @Test
    void toComplexTest() {
        aEntries = new double[][]
                {{1, 2, 0.912334, Double.NaN},
                {Double.POSITIVE_INFINITY, -9.322, -1992, 6434.2445}};
        A = new Matrix(aEntries);
        expEntries = new CNumber[][]
                {{new CNumber(1), new CNumber(2), new CNumber(0.912334), new CNumber(Double.NaN)},
                {new CNumber(Double.POSITIVE_INFINITY), new CNumber(-9.322), new CNumber(-1992), new CNumber(6434.2445)}};
        exp = new CMatrix(expEntries);
        act = A.toComplex();

        for(int i=0; i<exp.numRows(); i++) {
            for(int j=0; j<exp.numCols(); j++) {
                if(!Double.isNaN(exp.entries[i][j].re)) {
                    assertEquals(exp.entries[i][j], act.entries[i][j]);
                } else {
                    assertTrue(Double.isNaN(act.entries[i][j].re));
                    assertEquals(0, act.entries[i][j].im);
                }
            }
        }
    }
}

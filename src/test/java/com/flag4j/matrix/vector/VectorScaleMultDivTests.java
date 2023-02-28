package com.flag4j.matrix.vector;

import com.flag4j.CVector;
import com.flag4j.Vector;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class VectorScaleMultDivTests {

    double[] aEntries;
    Vector a;

    @Test
    void realScalMultTest() {
        double b;
        double[] expEntries;
        Vector exp;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new Vector(aEntries);
        b = -34.5;
        expEntries = new double[]{aEntries[0]*b, aEntries[1]*b, aEntries[2]*b};
        exp = new Vector(expEntries);

        assertEquals(exp, a.scalMult(b));
    }


    @Test
    void complexScalMultTest() {
        CNumber b;
        CNumber[] expEntries;
        CVector exp;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new Vector(aEntries);
        b = new CNumber(-0.99234, 1.56);
        expEntries = new CNumber[]{b.mult(aEntries[0]), b.mult(aEntries[1]), b.mult(aEntries[2])};
        exp = new CVector(expEntries);

        assertEquals(exp, a.scalMult(b));
    }


    // ---------------------------------------------------------------------------------------------------------------

    @Test
    void realScalDivTest() {
        double b;
        double[] expEntries;
        Vector exp;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new Vector(aEntries);
        b = -34.5;
        expEntries = new double[]{aEntries[0]/b, aEntries[1]/b, aEntries[2]/b};
        exp = new Vector(expEntries);

        assertEquals(exp, a.scalDiv(b));
    }


    @Test
    void complexScalDivTest() {
        CNumber b;
        CNumber[] expEntries;
        CVector exp;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new Vector(aEntries);
        b = new CNumber(-0.99234, 1.56);
        expEntries = new CNumber[]{new CNumber(aEntries[0]).div(b), new CNumber(aEntries[1]).div(b),
                new CNumber(aEntries[2]).div(b)};
        exp = new CVector(expEntries);

        assertEquals(exp, a.scalDiv(b));
    }
}

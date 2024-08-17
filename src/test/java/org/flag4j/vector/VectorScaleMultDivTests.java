package org.flag4j.vector;

import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.VectorOld;
import org.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class VectorScaleMultDivTests {

    double[] aEntries;
    VectorOld a;

    @Test
    void realScalMultTestCase() {
        double b;
        double[] expEntries;
        VectorOld exp;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new VectorOld(aEntries);
        b = -34.5;
        expEntries = new double[]{aEntries[0]*b, aEntries[1]*b, aEntries[2]*b};
        exp = new VectorOld(expEntries);

        assertEquals(exp, a.mult(b));
    }


    @Test
    void complexScalMultTestCase() {
        CNumber b;
        CNumber[] expEntries;
        CVectorOld exp;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new VectorOld(aEntries);
        b = new CNumber(-0.99234, 1.56);
        expEntries = new CNumber[]{b.mult(aEntries[0]), b.mult(aEntries[1]), b.mult(aEntries[2])};
        exp = new CVectorOld(expEntries);

        assertEquals(exp, a.mult(b));
    }


    // ---------------------------------------------------------------------------------------------------------------

    @Test
    void realScalDivTestCase() {
        double b;
        double[] expEntries;
        VectorOld exp;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new VectorOld(aEntries);
        b = -34.5;
        expEntries = new double[]{aEntries[0]/b, aEntries[1]/b, aEntries[2]/b};
        exp = new VectorOld(expEntries);

        assertEquals(exp, a.div(b));
    }


    @Test
    void complexScalDivTestCase() {
        CNumber b;
        CNumber[] expEntries;
        CVectorOld exp;

        // ------------------------ Sub-case 1 ------------------------
        aEntries = new double[]{1.234, -9.4, 8.45};
        a = new VectorOld(aEntries);
        b = new CNumber(-0.99234, 1.56);
        expEntries = new CNumber[]{new CNumber(aEntries[0]).div(b), new CNumber(aEntries[1]).div(b),
                new CNumber(aEntries[2]).div(b)};
        exp = new CVectorOld(expEntries);

        assertEquals(exp, a.div(b));
    }
}

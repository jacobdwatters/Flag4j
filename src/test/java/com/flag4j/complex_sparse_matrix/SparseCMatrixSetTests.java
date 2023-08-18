package com.flag4j.complex_sparse_matrix;

import com.flag4j.Shape;
import com.flag4j.SparseCMatrix;
import com.flag4j.complex_numbers.CNumber;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SparseCMatrixSetTests {

    @Test
    void setTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        SparseCMatrix a;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        SparseCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.12119+0.74369i"), new CNumber("0.92925+0.0336i"), new CNumber("0.96427+0.49577i"), new CNumber("0.053+0.9829i")};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{1, 1, 4, 1};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3, 5);
        expEntries = new CNumber[]{new CNumber("0.12119+0.74369i"), new CNumber("0.92925+0.0336i"), new CNumber("445.0"), new CNumber("0.96427+0.49577i"), new CNumber("0.053+0.9829i")};
        expRowIndices = new int[]{0, 1, 1, 1, 2};
        expColIndices = new int[]{1, 1, 2, 4, 1};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.set(445, 1, 2));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new CNumber[]{new CNumber("0.4921+0.29279i"), new CNumber("0.13959+0.33724i"), new CNumber("0.60995+0.257i"), new CNumber("0.74365+0.27311i"), new CNumber("0.91068+0.75977i"), new CNumber("0.25598+0.80711i"), new CNumber("0.69927+0.9189i"), new CNumber("0.88488+0.94923i"), new CNumber("0.18563+0.32057i"), new CNumber("0.14302+0.76377i"), new CNumber("0.12337+0.50506i"), new CNumber("0.5966+0.54101i"), new CNumber("0.28354+0.38644i"), new CNumber("0.40568+0.14458i"), new CNumber("0.3541+0.37016i"), new CNumber("0.62362+0.73473i"), new CNumber("0.54154+0.11671i"), new CNumber("0.26052+0.17949i"), new CNumber("0.58631+0.78973i"), new CNumber("0.87975+0.01091i"), new CNumber("0.95361+0.07703i"), new CNumber("0.39905+0.00734i"), new CNumber("0.35175+0.76207i"), new CNumber("0.97743+0.43951i"), new CNumber("0.71768+0.2805i")};
        aRowIndices = new int[]{0, 0, 2, 3, 3, 5, 5, 6, 6, 8, 10, 11, 11, 12, 12, 14, 16, 18, 19, 20, 21, 21, 21, 21, 22};
        aColIndices = new int[]{4, 9, 6, 7, 8, 4, 8, 3, 6, 7, 3, 1, 4, 1, 7, 8, 9, 2, 9, 1, 0, 2, 6, 7, 6};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(23, 11);
        expEntries = new CNumber[]{new CNumber("0.4921+0.29279i"), new CNumber("0.13959+0.33724i"), new CNumber("0.60995+0.257i"), new CNumber("0.74365+0.27311i"), new CNumber("0.91068+0.75977i"), new CNumber("0.25598+0.80711i"), new CNumber("0.69927+0.9189i"), new CNumber("0.88488+0.94923i"), new CNumber("0.18563+0.32057i"), new CNumber("0.14302+0.76377i"), new CNumber("0.12337+0.50506i"), new CNumber("0.5966+0.54101i"), new CNumber("0.28354+0.38644i"), new CNumber("0.40568+0.14458i"), new CNumber("0.3541+0.37016i"), new CNumber("0.62362+0.73473i"), new CNumber("-5.2"), new CNumber("0.54154+0.11671i"), new CNumber("0.26052+0.17949i"), new CNumber("0.58631+0.78973i"), new CNumber("0.87975+0.01091i"), new CNumber("0.95361+0.07703i"), new CNumber("0.39905+0.00734i"), new CNumber("0.35175+0.76207i"), new CNumber("0.97743+0.43951i"), new CNumber("0.71768+0.2805i")};
        expRowIndices = new int[]{0, 0, 2, 3, 3, 5, 5, 6, 6, 8, 10, 11, 11, 12, 12, 14, 15, 16, 18, 19, 20, 21, 21, 21, 21, 22};
        expColIndices = new int[]{4, 9, 6, 7, 8, 4, 8, 3, 6, 7, 3, 1, 4, 1, 7, 8, 9, 9, 2, 9, 1, 0, 2, 6, 7, 6};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.set(-5.2, 15, 9));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(1000, 32156);
        aEntries = new CNumber[]{new CNumber("0.76483+0.33276i"), new CNumber("0.98077+0.54744i"), new CNumber("0.39142+0.33343i"), new CNumber("0.31495+0.65704i"), new CNumber("0.18212+0.0667i"), new CNumber("0.77918+0.62381i"), new CNumber("0.21766+0.26696i"), new CNumber("0.20254+0.50045i"), new CNumber("0.38428+0.85978i")};
        aRowIndices = new int[]{107, 245, 282, 363, 619, 779, 794, 887, 988};
        aColIndices = new int[]{3196, 19199, 11012, 30674, 16995, 28494, 18961, 26823, 4341};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(1000, 32156);
        expEntries = new CNumber[]{new CNumber("0.76483+0.33276i"), new CNumber("7.2"), new CNumber("0.98077+0.54744i"), new CNumber("0.39142+0.33343i"), new CNumber("0.31495+0.65704i"), new CNumber("0.18212+0.0667i"), new CNumber("0.77918+0.62381i"), new CNumber("0.21766+0.26696i"), new CNumber("0.20254+0.50045i"), new CNumber("0.38428+0.85978i")};
        expRowIndices = new int[]{107, 234, 245, 282, 363, 619, 779, 794, 887, 988};
        expColIndices = new int[]{3196, 11002, 19199, 11012, 30674, 16995, 28494, 18961, 26823, 4341};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.set(7.2, 234, 11002));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(5, 2);
        aEntries = new CNumber[]{new CNumber("0.38871+0.73253i"), new CNumber("0.56447+0.3675i")};
        aRowIndices = new int[]{0, 2};
        aColIndices = new int[]{1, 1};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        SparseCMatrix final0a = a;
        assertThrows(Exception.class, ()->final0a.set(1, 6, 1));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(5, 2);
        aEntries = new CNumber[]{new CNumber("0.46549+0.31437i"), new CNumber("0.22736+0.59527i")};
        aRowIndices = new int[]{3, 4};
        aColIndices = new int[]{0, 1};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        SparseCMatrix final1a = a;
        assertThrows(Exception.class, ()->final1a.set(1, 1, 9));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(5, 2);
        aEntries = new CNumber[]{new CNumber("0.26673+0.90314i"), new CNumber("0.87254+0.61073i")};
        aRowIndices = new int[]{1, 4};
        aColIndices = new int[]{1, 1};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        SparseCMatrix final2a = a;
        assertThrows(Exception.class, ()->final2a.set(1, -1, 1));

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(5, 2);
        aEntries = new CNumber[]{new CNumber("0.38003+0.55457i"), new CNumber("0.31182+0.59758i")};
        aRowIndices = new int[]{2, 4};
        aColIndices = new int[]{1, 0};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        SparseCMatrix final3a = a;
        assertThrows(Exception.class, ()->final3a.set(1, 1, -1));

        // ---------------------  Sub-case 8 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.12119+0.74369i"), new CNumber("0.92925+0.0336i"), new CNumber("0.96427+0.49577i"), new CNumber("0.053+0.9829i")};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{1, 1, 4, 1};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3, 5);
        expEntries = new CNumber[]{new CNumber("0.12119+0.74369i"), new CNumber("0.92925+0.0336i"), new CNumber("0.96427+0.49577i"), new CNumber(34.8)};
        expRowIndices = new int[]{0, 1, 1, 2};
        expColIndices = new int[]{1, 1, 4, 1};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.set(34.8, 2, 1));
    }


    @Test
    void setComplexTest() {
        Shape aShape;
        int[] aRowIndices;
        int[] aColIndices;
        CNumber[] aEntries;
        SparseCMatrix a;

        Shape expShape;
        int[] expRowIndices;
        int[] expColIndices;
        CNumber[] expEntries;
        SparseCMatrix exp;

        // ---------------------  Sub-case 1 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.12119+0.74369i"), new CNumber("0.92925+0.0336i"), new CNumber("0.96427+0.49577i"), new CNumber("0.053+0.9829i")};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{1, 1, 4, 1};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3, 5);
        expEntries = new CNumber[]{new CNumber("0.12119+0.74369i"), new CNumber("0.92925+0.0336i"), new CNumber(13, 4.99), new CNumber("0.96427+0.49577i"), new CNumber("0.053+0.9829i")};
        expRowIndices = new int[]{0, 1, 1, 1, 2};
        expColIndices = new int[]{1, 1, 2, 4, 1};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.set(new CNumber(13, 4.99), 1, 2));

        // ---------------------  Sub-case 2 ---------------------
        aShape = new Shape(23, 11);
        aEntries = new CNumber[]{new CNumber("0.4921+0.29279i"), new CNumber("0.13959+0.33724i"), new CNumber("0.60995+0.257i"), new CNumber("0.74365+0.27311i"), new CNumber("0.91068+0.75977i"), new CNumber("0.25598+0.80711i"), new CNumber("0.69927+0.9189i"), new CNumber("0.88488+0.94923i"), new CNumber("0.18563+0.32057i"), new CNumber("0.14302+0.76377i"), new CNumber("0.12337+0.50506i"), new CNumber("0.5966+0.54101i"), new CNumber("0.28354+0.38644i"), new CNumber("0.40568+0.14458i"), new CNumber("0.3541+0.37016i"), new CNumber("0.62362+0.73473i"), new CNumber("0.54154+0.11671i"), new CNumber("0.26052+0.17949i"), new CNumber("0.58631+0.78973i"), new CNumber("0.87975+0.01091i"), new CNumber("0.95361+0.07703i"), new CNumber("0.39905+0.00734i"), new CNumber("0.35175+0.76207i"), new CNumber("0.97743+0.43951i"), new CNumber("0.71768+0.2805i")};
        aRowIndices = new int[]{0, 0, 2, 3, 3, 5, 5, 6, 6, 8, 10, 11, 11, 12, 12, 14, 16, 18, 19, 20, 21, 21, 21, 21, 22};
        aColIndices = new int[]{4, 9, 6, 7, 8, 4, 8, 3, 6, 7, 3, 1, 4, 1, 7, 8, 9, 2, 9, 1, 0, 2, 6, 7, 6};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(23, 11);
        expEntries = new CNumber[]{new CNumber("0.4921+0.29279i"), new CNumber("0.13959+0.33724i"), new CNumber("0.60995+0.257i"), new CNumber("0.74365+0.27311i"), new CNumber("0.91068+0.75977i"), new CNumber("0.25598+0.80711i"), new CNumber("0.69927+0.9189i"), new CNumber("0.88488+0.94923i"), new CNumber("0.18563+0.32057i"), new CNumber("0.14302+0.76377i"), new CNumber("0.12337+0.50506i"), new CNumber("0.5966+0.54101i"), new CNumber("0.28354+0.38644i"), new CNumber("0.40568+0.14458i"), new CNumber("0.3541+0.37016i"), new CNumber("0.62362+0.73473i"), new CNumber(0, 24.1), new CNumber("0.54154+0.11671i"), new CNumber("0.26052+0.17949i"), new CNumber("0.58631+0.78973i"), new CNumber("0.87975+0.01091i"), new CNumber("0.95361+0.07703i"), new CNumber("0.39905+0.00734i"), new CNumber("0.35175+0.76207i"), new CNumber("0.97743+0.43951i"), new CNumber("0.71768+0.2805i")};
        expRowIndices = new int[]{0, 0, 2, 3, 3, 5, 5, 6, 6, 8, 10, 11, 11, 12, 12, 14, 15, 16, 18, 19, 20, 21, 21, 21, 21, 22};
        expColIndices = new int[]{4, 9, 6, 7, 8, 4, 8, 3, 6, 7, 3, 1, 4, 1, 7, 8, 9, 9, 2, 9, 1, 0, 2, 6, 7, 6};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.set(new CNumber(0, 24.1), 15, 9));

        // ---------------------  Sub-case 3 ---------------------
        aShape = new Shape(1000, 32156);
        aEntries = new CNumber[]{new CNumber("0.76483+0.33276i"), new CNumber("0.98077+0.54744i"), new CNumber("0.39142+0.33343i"), new CNumber("0.31495+0.65704i"), new CNumber("0.18212+0.0667i"), new CNumber("0.77918+0.62381i"), new CNumber("0.21766+0.26696i"), new CNumber("0.20254+0.50045i"), new CNumber("0.38428+0.85978i")};
        aRowIndices = new int[]{107, 245, 282, 363, 619, 779, 794, 887, 988};
        aColIndices = new int[]{3196, 19199, 11012, 30674, 16995, 28494, 18961, 26823, 4341};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(1000, 32156);
        expEntries = new CNumber[]{new CNumber("0.76483+0.33276i"), new CNumber(-1.4, -99.2), new CNumber("0.98077+0.54744i"), new CNumber("0.39142+0.33343i"), new CNumber("0.31495+0.65704i"), new CNumber("0.18212+0.0667i"), new CNumber("0.77918+0.62381i"), new CNumber("0.21766+0.26696i"), new CNumber("0.20254+0.50045i"), new CNumber("0.38428+0.85978i")};
        expRowIndices = new int[]{107, 234, 245, 282, 363, 619, 779, 794, 887, 988};
        expColIndices = new int[]{3196, 11002, 19199, 11012, 30674, 16995, 28494, 18961, 26823, 4341};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.set(new CNumber(-1.4, -99.2), 234, 11002));

        // ---------------------  Sub-case 4 ---------------------
        aShape = new Shape(5, 2);
        aEntries = new CNumber[]{new CNumber("0.38871+0.73253i"), new CNumber("0.56447+0.3675i")};
        aRowIndices = new int[]{0, 2};
        aColIndices = new int[]{1, 1};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        SparseCMatrix final0a = a;
        assertThrows(Exception.class, ()->final0a.set(1, 6, 1));

        // ---------------------  Sub-case 5 ---------------------
        aShape = new Shape(5, 2);
        aEntries = new CNumber[]{new CNumber("0.46549+0.31437i"), new CNumber("0.22736+0.59527i")};
        aRowIndices = new int[]{3, 4};
        aColIndices = new int[]{0, 1};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        SparseCMatrix final1a = a;
        assertThrows(Exception.class, ()->final1a.set(new CNumber(1, 1), 1, 9));

        // ---------------------  Sub-case 6 ---------------------
        aShape = new Shape(5, 2);
        aEntries = new CNumber[]{new CNumber("0.26673+0.90314i"), new CNumber("0.87254+0.61073i")};
        aRowIndices = new int[]{1, 4};
        aColIndices = new int[]{1, 1};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        SparseCMatrix final2a = a;
        assertThrows(Exception.class, ()->final2a.set(new CNumber(1, 1), -1, 1));

        // ---------------------  Sub-case 7 ---------------------
        aShape = new Shape(5, 2);
        aEntries = new CNumber[]{new CNumber("0.38003+0.55457i"), new CNumber("0.31182+0.59758i")};
        aRowIndices = new int[]{2, 4};
        aColIndices = new int[]{1, 0};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        SparseCMatrix final3a = a;
        assertThrows(Exception.class, ()->final3a.set(new CNumber(1, 1), 1, -1));

        // ---------------------  Sub-case 8 ---------------------
        aShape = new Shape(3, 5);
        aEntries = new CNumber[]{new CNumber("0.12119+0.74369i"), new CNumber("0.92925+0.0336i"), new CNumber("0.96427+0.49577i"), new CNumber("0.053+0.9829i")};
        aRowIndices = new int[]{0, 1, 1, 2};
        aColIndices = new int[]{1, 1, 4, 1};
        a = new SparseCMatrix(aShape, aEntries, aRowIndices, aColIndices);

        expShape = new Shape(3, 5);
        expEntries = new CNumber[]{new CNumber("0.12119+0.74369i"), new CNumber("0.92925+0.0336i"), new CNumber("0.96427+0.49577i"), new CNumber(13, 4.99)};
        expRowIndices = new int[]{0, 1, 1, 2};
        expColIndices = new int[]{1, 1, 4, 1};
        exp = new SparseCMatrix(expShape, expEntries, expRowIndices, expColIndices);

        assertEquals(exp, a.set(new CNumber(13, 4.99), 2, 1));
    }
}

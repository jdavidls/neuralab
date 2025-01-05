# Contratos de Futuros en Binance: Formulaciones y Reglas

## Introducción
Este documento detalla las fórmulas matemáticas y las reglas fundamentales que sustentan el funcionamiento de los contratos de futuros ofrecidos en Binance. La información cubre tanto contratos USDⓇ-M como COIN-ⓇM, incluyendo aspectos como ganancias y pérdidas, margen requerido, precio de liquidación, y comisiones de financiación.

---

## 1. Ganancias y Pérdidas No Realizadas (PnL)

### 1.1 Contratos USDⓇ-M

#### Posición larga (long):
\[
\text{PnL}_{\text{long}} = Q \cdot (P_{\text{mark}} - P_{\text{entry}})
\]

#### Posición corta (short):
\[
\text{PnL}_{\text{short}} = Q \cdot (P_{\text{entry}} - P_{\text{mark}})
\]

**Donde:**
- \( Q \): Tamaño de la posición (en contratos).
- \( P_{\text{mark}} \): Precio de marca.
- \( P_{\text{entry}} \): Precio de entrada.

### 1.2 Contratos COIN-ⓇM

#### Fórmula general:
\[
\text{PnL} = Q \cdot V \cdot \left( \frac{1}{P_{\text{entry}}} - \frac{1}{P_{\text{exit}}} \right)
\]

**Donde:**
- \( V \): Valor del contrato (en criptoactivos).
- \( P_{\text{exit}} \): Precio de salida.

---

## 2. Margen Requerido

### 2.1 Modo Unidireccional
\[
M_{\text{req}} = \max \left( |N_{\text{pos}} + N_{\text{buy}}|, |N_{\text{pos}} - N_{\text{sell}}| \right) \cdot \frac{1}{L}
\]

### 2.2 Modo de Cobertura
\[
M_{\text{req}} = M_{\text{long}} + M_{\text{short}}
\]

#### Cálculo de componentes:
\[
M_{\text{long}} = \max \left( |N_{\text{pos}}^{\text{long}} + N_{\text{buy}}^{\text{long}}|, |N_{\text{pos}}^{\text{long}} - N_{\text{sell}}^{\text{long}}| \right) \cdot \frac{1}{L}
\]
\[
M_{\text{short}} = \max \left( |N_{\text{pos}}^{\text{short}} + N_{\text{buy}}^{\text{short}}|, |N_{\text{pos}}^{\text{short}} - N_{\text{sell}}^{\text{short}}| \right) \cdot \frac{1}{L}
\]

**Donde:**
- \( N_{\text{pos}} \): Valor nocional de la posición.
- \( N_{\text{buy}} \), \( N_{\text{sell}} \): Valores nocionales de órdenes de compra o venta.
- \( L \): Apalancamiento.

---

## 3. Precio de Liquidación

### 3.1 Contratos USDⓇ-M
\[
P_{\text{liq}} = \frac{M_{\text{total}} \cdot L}{M_{\text{total}} - M_{\text{maint}} \cdot L}
\]

### 3.2 Contratos COIN-ⓇM
\[
P_{\text{liq}} = \frac{V \cdot Q}{\frac{M_{\text{total}}}{L} - M_{\text{maint}}}
\]

**Donde:**
- \( M_{\text{total}} \): Margen total.
- \( M_{\text{maint}} \): Margen de mantenimiento.
- \( V \): Valor del contrato.

---

## 4. Precio de Marca

\[
P_{\text{mark}} = P_{\text{index}} + \text{Premium Index}
\]

**Donde:**
- \( P_{\text{index}} \): Precio del índice basado en mercados spot.
- \( \text{Premium Index} \): Diferencia entre precio de futuros y spot.

---

## 5. Comisión de Financiación (Funding Rate)

\[
\text{Pago de Financiación} = Q \cdot \text{Funding Rate}
\]

**Funding Rate:**
\[
\text{Funding Rate} = \text{Clamp} \left( \text{Premium Index} + \text{Interest Rate}, -\text{Cap}, \text{Cap} \right)
\]

**Donde:**
- \( \text{Clamp} \): Función que limita el rango de la tasa.
- \( \text{Interest Rate} \): Tasa de interés diaria.
- \( \text{Cap} \): Valor máximo permitido para la tasa.

---

## 6. Margen Inicial y Margen de Mantenimiento

### Margen Inicial
\[
M_{\text{inicial}} = \frac{V_{\text{nocional}}}{L}
\]

### Margen de Mantenimiento
\[
M_{\text{mantenimiento}} = V_{\text{nocional}} \cdot \text{Tasa}_{\text{mantenimiento}}
\]

**Donde:**
- \( V_{\text{nocional}} \): Valor nocional de la posición.
- \( \text{Tasa}_{\text{mantenimiento}} \): Tasa establecida por Binance.

---

## Reglas Fundamentales
1. **Apalancamiento:** Define el riesgo y el margen requerido; puede ajustarse según el activo y el contrato.
2. **Precio de Marca:** Protege contra liquidaciones indebidas durante alta volatilidad.
3. **Margen de Mantenimiento:** Evita liquidaciones automáticas si el capital cae por debajo del umbral.
4. **Comisión de Financiación:** Balancea precios de futuros y spot mediante pagos entre traders.

---

**Nota:** Consulte siempre las especificaciones oficiales de Binance para garantizar información actualizada y precisa.


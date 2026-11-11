## Diagramas de teia (cobweb plots) usados na previa da dissertacao.
##
## Gera as duas figuras do mapa linear y[n+1] = a*y[n] (filtro IIR de
## primeira ordem sem entrada), ambas com condicao inicial y[0] = 0,64:
##   coweb_reta_estavel.png   -> a = 0,5 (ponto fixo y* = 0 estavel)
##   coweb_reta_instavel.png  -> a = 1,2 (ponto fixo y* = 0 instavel)
## e salva os arquivos em "Previa Dissertacao/Imagens". Como o LaTeX procura
## .png antes de .jpg, depois de gerar as figuras os antigos
## coweb_reta_*.jpg podem ser apagados.
##
## Para salvar em outra pasta, defina out_dir antes de rodar o script:
##   out_dir = "C:/alguma/pasta"; run("cobweb_plot.m")
##
## Fonte: o texto usa "Latin Modern Roman" (versao OpenType da Computer Modern,
## a fonte padrao do LaTeX). O Octave so encontra fontes instaladas no sistema;
## para instalar no Windows, clique com o botao direito em
##   C:\Program Files\MiKTeX\fonts\opentype\public\lm\lmroman10-regular.otf
## e escolha "Instalar para todos os usuarios". Sem a fonte instalada, o
## Octave usa a fonte padrao dele.
##
## A funcao plot_cobweb aceita qualquer mapa f, entao tambem serve para
## explorar o mapa logistico, por exemplo:
##   plot_cobweb (@(x) 3.3*x.*(1 - x), 0.64, 100, [0 1], [0 1], ...
##                "x[n]", "x[n+1]", "Mapa logístico, r = 3,3", "", "")

1;  # marca o arquivo como script (permite definir funcoes abaixo)

FONT_NAME = "Latin Modern Roman";
FONT_SIZE = 16;          # pt; a figura de 6 pol. vai para 0.5\textwidth (~0.55x)

set (0, "defaultaxesfontname", FONT_NAME);
set (0, "defaulttextfontname", FONT_NAME);
set (0, "defaultaxesfontsize", FONT_SIZE);
set (0, "defaulttextfontsize", FONT_SIZE);

function fig = plot_cobweb (f, y0, n_iter, x_lim, y_lim, x_label, y_label, ...
                            titulo, legenda_f, arquivo)
  ## Desenha o diagrama de teia do mapa f a partir de y0 e, se arquivo nao for
  ## vazio, salva a figura em PNG.
  fig = figure ();
  clf;
  hold on;

  x_vals = linspace (x_lim(1), x_lim(2), 500);
  h_f = plot (x_vals, f(x_vals), "b-", "LineWidth", 2);
  h_id = plot (x_vals, x_vals, "r--", "LineWidth", 1.5);

  ## Trajetoria: sobe ate a curva, vai ate a bissetriz e repete
  y_cur = y0;
  h_traj = plot ([y_cur, y_cur], [0, f(y_cur)], "-", ...
                 "Color", [0 0.6 0], "LineWidth", 1.5);
  for i = 1:n_iter
    y_next = f(y_cur);
    plot ([y_cur, y_next], [y_next, y_next], "-", "Color", [0 0.6 0], "LineWidth", 1.5);
    plot ([y_next, y_next], [y_next, f(y_next)], "-", "Color", [0 0.6 0], "LineWidth", 1.5);
    y_cur = y_next;
    if (y_cur > x_lim(2))   # trajetoria saiu da janela (caso instavel)
      break;
    endif
  endfor
  plot (y0, 0, "o", "Color", [0 0.6 0], "MarkerFaceColor", [0 0.6 0], "MarkerSize", 6);

  xlabel (x_label);
  ylabel (y_label);
  title (titulo, "FontWeight", "normal");
  if (! isempty (legenda_f))
    legend ([h_f, h_id, h_traj], {legenda_f, "y[n+1] = y[n]", "Trajetória"}, ...
            "location", "northwest", "FontSize", 14);
  endif
  xlim (x_lim);
  ylim (y_lim);
  grid on;
  box on;
  hold off;

  ## Virgula como separador decimal nos eixos (padrao pt-BR)
  ax = gca ();
  fmt = @(v) strrep (arrayfun (@(t) sprintf ("%g", t), v, "UniformOutput", false), ".", ",");
  set (ax, "xticklabel", fmt (get (ax, "xtick")));
  set (ax, "yticklabel", fmt (get (ax, "ytick")));

  if (! isempty (arquivo))
    ## 6 x 4,5 polegadas a 300 dpi
    set (fig, "paperunits", "inches", "paperposition", [0 0 6 4.5]);
    print (fig, arquivo, "-dpng", "-r300");
    printf ("Figura salva em %s\n", arquivo);
  endif
endfunction

if (! exist ("out_dir", "var"))
  out_dir = fullfile (fileparts (mfilename ("fullpath")), "..", ...
                      "Previa Dissertacao", "Imagens");
endif

y0 = 0.64;

## a = 0,5: a trajetoria converge para o ponto fixo estavel y* = 0
plot_cobweb (@(y) 0.5*y, y0, 30, [0 0.7], [0 0.7], "y[n]", "y[n+1]", ...
             "Diagrama de teia: a = 0,5", "f(y) = 0,5y", ...
             fullfile (out_dir, "coweb_reta_estavel.png"));

## a = 1,2: a trajetoria se afasta do ponto fixo instavel y* = 0
plot_cobweb (@(y) 1.2*y, y0, 30, [0 10], [0 12], "y[n]", "y[n+1]", ...
             "Diagrama de teia: a = 1,2", "f(y) = 1,2y", ...
             fullfile (out_dir, "coweb_reta_instavel.png"));

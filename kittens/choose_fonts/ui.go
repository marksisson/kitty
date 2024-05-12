package choose_fonts

import (
	"fmt"
	"strconv"
	"sync"

	"kitty/tools/tui"
	"kitty/tools/tui/graphics"
	"kitty/tools/tui/loop"
)

var _ = fmt.Print

type State int

const (
	SCANNING_FAMILIES State = iota
	LISTING_FAMILIES
	CHOOSING_FACES
)

type TextStyle struct {
	Font_sz    float64 `json:"font_size"`
	Dpi_x      float64 `json:"dpi_x"`
	Dpi_y      float64 `json:"dpi_y"`
	Foreground string  `json:"foreground"`
	Background string  `json:"background"`
}

type handler struct {
	lp                   *loop.Loop
	state                State
	err_mutex            sync.Mutex
	err_in_worker_thread error
	mouse_state          tui.MouseState
	render_count         uint
	render_lines         tui.RenderLines
	text_style           TextStyle
	graphics_manager     graphics_manager

	listing FontList
}

func (h *handler) set_worker_error(err error) {
	h.err_mutex.Lock()
	defer h.err_mutex.Unlock()
	h.err_in_worker_thread = err
}

func (h *handler) get_worker_error() error {
	h.err_mutex.Lock()
	defer h.err_mutex.Unlock()
	return h.err_in_worker_thread
}

// Events {{{
func (h *handler) initialize() {
	h.lp.SetCursorVisible(false)
	h.lp.OnQueryResponse = h.on_query_response
	h.lp.QueryTerminal("font_size", "dpi_x", "dpi_y", "foreground", "background")
	h.listing.initialize(h)
	h.draw_screen()
	initialize_variable_data_cache()
	h.graphics_manager.initialize(h.lp)
	go func() {
		var r ListResult
		h.set_worker_error(kitty_font_backend.query("list_monospaced_fonts", nil, &r))
		h.listing.fonts = r.Fonts
		h.listing.resolved_faces_from_kitty_conf = r.Resolved_faces
		h.lp.WakeupMainThread()
	}()
}

func (h *handler) finalize() {
	h.lp.SetCursorVisible(true)
	h.lp.SetCursorShape(loop.BLOCK_CURSOR, true)
	h.graphics_manager.finalize(h.lp)
}

func (h *handler) on_query_response(key, val string, valid bool) error {
	if !valid {
		return fmt.Errorf("Terminal does not support querying the: %s", key)
	}
	set_float := func(k, v string, dest *float64) error {
		if fs, err := strconv.ParseFloat(v, 64); err == nil {
			*dest = fs
		} else {
			return fmt.Errorf("Invalid response from terminal to %s query: %#v", k, v)
		}
		return nil
	}
	switch key {
	case "font_size":
		if err := set_float(key, val, &h.text_style.Font_sz); err != nil {
			return err
		}
	case "dpi_x":
		if err := set_float(key, val, &h.text_style.Dpi_x); err != nil {
			return err
		}
	case "dpi_y":
		if err := set_float(key, val, &h.text_style.Dpi_y); err != nil {
			return err
		}
	case "foreground":
		h.text_style.Foreground = val
	case "background":
		h.text_style.Background = val
	}
	return nil
}

func (h *handler) draw_screen() (err error) {
	h.render_count++
	h.lp.StartAtomicUpdate()
	defer func() {
		h.mouse_state.UpdateHoveredIds()
		h.mouse_state.ApplyHoverStyles(h.lp)
		h.lp.EndAtomicUpdate()
	}()
	h.lp.ClearScreen()
	h.lp.AllowLineWrapping(false)
	h.mouse_state.ClearCellRegions()
	switch h.state {
	case SCANNING_FAMILIES:
		h.lp.Println("Scanning system for fonts, please wait...")
	case LISTING_FAMILIES:
		return h.listing.draw_screen()
	}
	return
}

func (h *handler) on_wakeup() (err error) {
	if err = h.get_worker_error(); err != nil {
		return
	}
	switch h.state {
	case SCANNING_FAMILIES:
		h.state = LISTING_FAMILIES
		h.listing.on_wakeup()
	}
	return h.draw_screen()
}

func (h *handler) on_mouse_event(event *loop.MouseEvent) (err error) {
	rc := h.render_count
	redraw_needed := false
	if h.mouse_state.UpdateState(event) {
		redraw_needed = true
	}
	if event.Event_type == loop.MOUSE_CLICK && event.Buttons&loop.LEFT_MOUSE_BUTTON != 0 {
		if err = h.mouse_state.ClickHoveredRegions(); err != nil {
			return
		}
	}
	if redraw_needed && rc == h.render_count {
		err = h.draw_screen()
	}
	return
}

func (h *handler) on_key_event(event *loop.KeyEvent) (err error) {
	if event.MatchesPressOrRepeat("ctrl+c") {
		event.Handled = true
		return fmt.Errorf("canceled by user")
	}
	switch h.state {
	case LISTING_FAMILIES:
		return h.listing.on_key_event(event)
	}
	return
}

func (h *handler) on_text(text string, from_key_event bool, in_bracketed_paste bool) (err error) {
	switch h.state {
	case LISTING_FAMILIES:
		return h.listing.on_text(text, from_key_event, in_bracketed_paste)
	}
	return
}

func (h *handler) on_escape_code(etype loop.EscapeCodeType, payload []byte) error {
	switch etype {
	case loop.APC:
		gc := graphics.GraphicsCommandFromAPC(payload)
		if gc != nil {
			return h.graphics_manager.on_response(gc)
		}
	}
	return nil
}

// }}}